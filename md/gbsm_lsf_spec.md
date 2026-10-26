# Spec: port of the QuaDRiGa large-scale model to quadriga-lib

Status: **draft r1** (revision after review round 1, 2026-10-26; r0 by the spec writer, same day), to be reviewed
in three rounds; findings go to `md/gbsm_lsf_findings.md`; the changes against r0 are listed in section 12. Branch
`fixes` of `/home/stephan/sjc/quadriga-lib-fixes`, base cebd0c0 (v0.12.3).

This document replaces `md/gbsm_lsf_plan.md` as the single source of truth for the implementation. Everything an
implementation agent needs is here or referenced by file and line. Markers:

- `[Dn]` cites decision n of section 1.2 (carried over from the plan; binding unless this spec states a better
  solution together with its reason, as allowed by Stephan's review of r0).
- `[W-n]` is a choice of the spec writer where the plan was silent (list in section 1.4). Reviewers check them.
- `[I-n]` is a problem with the plan found while writing (list in section 1.5). The decision is kept; the issue is
  stated for Stephan.
- `[pending user decision R1-nn]` marks text that stays as in r0 until Stephan decides finding R1-nn; the note at
  the place says which local edit applies the alternative.
- None of these markers, no ID of the findings, no section number of this spec or the plan, no `md/` path and no
  QuaDRiGa-NG line number may appear in any produced source, header, test, wrapper or conf file (9.1).

Process change against the SPEC files: the review points A, B and C of `md/SPEC_CPP_API.md`,
`md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md` are **not** stops for a human. The implementation agent decides
within this spec, never against it, and records every decision and every deviation in its run report (section 7).
Where the spec and a SPEC file disagree, this spec wins; the known differences are listed in 1.6.

Contents

1. Scope and decisions
2. Public API (`include/quadriga_gbsm.hpp`)
3. Normative behaviour (shared definitions, the six new functions, the moved functions)
4. Conf-key changes and the 23 v19 files
5. Catch2 tests
6. MEX and Python wrappers and their tests
7. Implementation runs R1, R2a, R2b, R3a, R3b, R4a, R4b
8. Requirements from the later SSF port
9. Constraints (copied into every run)
10. Integration hand-over (QuaDRiGa-NG, after the merge)
11. Appendix: tables
12. Change log


## 1. Scope and decisions

### 1.1 Goal

Port the large-scale part of QuaDRiGa (QuaDRiGa-NG `quadriga_src`) to the C++ core of quadriga-lib, with MEX and
Python wrappers:

- path loss and shadow-fading standard deviation (`qd_builder.get_pl`),
- the large-scale parameters (LSPs) DS, KF, SF, ASD, ASA, ESD, ESA, XPR (`qd_builder.lsp_vals`, the LSF part of
  `init_sos`, `gen_lsf_parameters`, `get_sf_profile`, `get_lsp_map`),
- the quantities drawn together with them: absolute TOA offset, ground-reflection permittivity, EoD offset,
- LOS probability and soft LOS state (probability part of `qd_layout.set_scenario`),
- O2I penetration loss (`qd_layout.gen_o2i_loss`).

Goal 1: full coverage of the large-scale model of 3GPP TR 38.901 v19.5.0 (clauses 7.4, 7.5 steps 2 to 4, 7.6.3.3,
7.6.9 and the LSF-related parts of 7.6.5, 7.6.10). Goal 2: QuaDRiGa-NG replaces its MATLAB implementations by
library calls (section 10).

Ten functions in the new public header `include/quadriga_gbsm.hpp`: four moved ones (`conf_file_read`,
`scenpar_validate`, `conf_file_list`, `conf_file_write`) and six new ones (`lsf_init_phases`, `lsf_draw`,
`path_loss`, `lsp_generate`, `los_probability`, `o2i_loss`).

Data flow for one builder (one BS site, one scenario):

```
conf_file_read(fn, scenpar, ..., &pl_model, &plpar)                     once per scenario file
lsf_init_phases(scenpar, phase, seed, &rx_pos, &tx_pos)                once per builder; the builder stores 'phase'
lsf_draw(scenpar, phase, tx_pos, rx_pos, rnd)                          normalized random values per link, [n_link, 11]
path_loss(pl_model, plpar, scenpar, fc, tx_pos, rx_pos, pl,
          &sf_sigma, nullptr, &rnd_he)                                  rnd_he = rnd.col(10); d3d_in only for legacy
lsp_generate(scenpar, fc, tx_pos, rx_pos, rnd, lsp, &sf_sigma, ...)    final LSPs per link and frequency
```

Layout level (before builders exist), in this order per drop:

```
o2i_loss(fc, rnd_o2i, loss, models, weights, d2d_in_max, &d2d_in)      indoor UTs: penetration loss and d2D-in
d2d_out = max(d2D - d2d_in, 0)                                          caller; outdoor UTs: d2d_in = 0
los_probability(model, d2d_out, h_bs, h_ut, p_los, ...)                 TR Table 7.4.2-1 uses d2D-out
```

`o2i_loss` runs before `los_probability` because Table 7.4.2-1 evaluates the LOS probability of an O2I UT at
d2D-out = d2D - d2D-in; the caller clamps d2D-out at 0 (`los_probability` rejects negative distances). Their random
inputs come from `sos_rand` / `sos_randn` at the TR correlation distances (Table 7.6.3.1-2, clause 7.6.3.3); they
are evaluated once per drop, so no phases are stored. Every model function is deterministic: randomness enters
only through normalized random values (N(0,1) or U(0,1)) passed by the caller.

### 1.2 Decisions (binding)

Answers given by Stephan on 2026-10-23, carried over from the plan unchanged. They are not reopened here.

| # | Question | Answer |
|---|---|---|
| D1 | Scope besides PL, SF and the 8 LSPs | All of: LOS probability and soft LOS state; O2I penetration loss; absolute TOA and ground-reflection permittivity; EoD offset and the new ESD terms. |
| D2 | Path-loss representation | Port the plpar models and extend them. All 10 parametric models of `get_pl.m` 1:1; the missing 38.901 behaviours as new `PL_` keys. |
| D3 | Random state | Phase arrays in and out. Free functions; the builder stores plain arrays; re-evaluation along tracks and on maps works with the stored phases. |
| D4 | Policy where TR and QuaDRiGa differ | Both supported, 3GPP behaviour is the default. QuaDRiGa-NG passes legacy flags where it wants the old behaviour. |
| D5 | Units of the outputs | As the builder: DS in s, KF, SF and XPR as linear power ratios, AS and ES in degrees, PL in dB. Arguments in degrees carry `_deg` in their name or say so in the comment. Approved exception to SPEC_CPP_API. |
| D6 | Spatial correlation of LSPs | 2D (horizontal plane) plus an optional floor ID; different floors are uncorrelated (Table 7.5-6 note 3). 3D (legacy) is an option. |
| D7 | LOS probability models | Named models in C++. Model parameters are function arguments with the TR values as defaults. The group table (group name to model and member files) stays in QuaDRiGa-NG `set_scenario`, reduced to names. |
| D7a | Conf format | Fixed. No new file type, no new string keys, no change of the reader and writer interfaces. Adding numeric keys to the key table and new `PL_` keys is allowed. |
| D8 | Group-level numeric settings | LOS-state and indoor-state decorrelation distances and the per-scenario O2I constants are numeric scenpar keys in the files where they are unique (NLOS files, O2I files). InF and SMa geometry (d_clutter, r, h_c, vegetation) are arguments of the named LOS models. |
| D9 | Input of the scenario | `scenpar` and `plpar` maps, exactly as `conf_file_read` returns them. |
| D10 | Header | New public header `include/quadriga_gbsm.hpp`. `conf_file_read`, `scenpar_validate`, `conf_file_list`, `conf_file_write` move there from `quadriga_tools.hpp` (clean move; `quadriga_lib.hpp` includes the new header). Later SSF and other QuaDRiGa-NG functions go there too. |
| D11 | Documentation section | "Geometry-based stochastic model", for the new functions and the four moved ones (they leave "Channel generation functions"). |
| D12 | Numeric type | `double` only, no template. |
| D13 | ACF of the LSP generators | `Comb256` default (QuaDRiGa); the preset is an argument, `Exp256` gives the literal TR ACF. |
| D14 | Bias removal of the LSP phases | Optional, on by default whenever drop positions are passed. |
| D15 | UMa random environment height hE | Spatially consistent SOS uniform variable, generator 10 of the LSF phase set, decorrelation distance `SF_lambda` (D19). |
| D16 | SF standard deviation | `path_loss` returns `sf_sigma` per link and frequency; `lsp_generate` takes it as optional input; the returned SF is final. Replaces `scale_sf` in `get_channels`. |
| D17 | Function set and names | `lsf_init_phases`, `lsf_draw`, `path_loss`, `lsp_generate`, `los_probability`, `o2i_loss`. |
| D18 | LSP output layout | One cube `[n_link, n_freq, 8]`, slices DS, KF, SF, ASD, ASA, ESD, ESA, XPR. |
| D19 | Decorrelation distance of hE | `SF_lambda`, no new key. |
| D20 | Dual mobility | Explicit `bool dual_mobility` argument. |
| D21 | Bias removal and reciprocity | Mirror first, then remove the bias with both devices. Fixes the quirk of the MATLAB code (bias removed before mirroring); dual-mobility values change against today. |
| D22 | Absolute TOA | Keep the rule "active if `absTOA_mu > -30`"; `abs_toa` = 0 otherwise; bound 2L/c when `absTOA_Lmax > 0`. |
| D23 | Legacy parity | Yes, in the integration phase: with legacy flags and the same phases the library reproduces today's MATLAB values to 1e-12 relative, single mobility only. The reference values come from a MATLAB script in NG; nothing in quadriga-lib depends on NG. |
| D24 | Renormalisation of the LOS-state draw | Not in the library. NG may keep it as a legacy option. |
| D25 | O2I scope | All of: d3D-in geometry, mmMAGIC model, car loss, < 6 GHz model. `gen_o2i_loss.m` can be deleted after the integration. |
| D26 | InF delay spread | Keys plus override: `DS_VS_a`, `DS_VS_b`, `DS_VS` in the InF files; argument `vs_ratio` of `lsp_generate` overrides `DS_VS`. |
| D27 | Optional TR NLOS path-loss formulas | Documented only (comments in the v19 NLOS files give the `logdist` parameters); no new files. |
| D28 | Review points A, B, C | Manager decides within the plan; escalates contract changes, deviations from a decision and open TR questions. In this spec process: the implementation agent decides within the spec and reports (see the header). |
| D29 | Python | New submodule `quadriga_lib.gbsm`; the four conf functions move there from `tools` (breaking for Python users). MATLAB names stay flat and unchanged. |
| D30 | Workers | The manager writes each worker prompt; Stephan starts the worker session. |
| D31 | Commits | Stephan commits; agents never commit. |
| D32 | Integration | Same manager, after the merge; NG becomes writable then. |

Departures from the wording of a decision in r1, each with its reason (Stephan's review of r0: decisions are not
binding where a better solution exists):
- D2 ("1:1"): `winner_los` uses `C2` beyond the breakpoint and per-link breakpoint heights [W-42, W-32] (MATLAB bug
  and call-dependent breakpoint); `dual_slope` gets `PL_BP2D` so that RMa/SMa follow the TR [W-41].
- D4 ("both supported"): no legacy flag for the TR 36.873 UMa `d^2` (a bug of `set_scenario.m`, R1-12); the legacy
  O2I distance mode is selected by passing `d3d_in` instead of a separate flag [W-46].
- D7 (named models): the two 36.873 names are dropped because they equal the 38.901 models (R1-12).
- D14 (bias removal by default): unchanged, but documented to act on one floor [W-43].
- D21 ("mirror first, then remove the bias with both devices"): realized as one correction over the union of
  both devices' positions, followed by the mirroring of the corrected phases [W-5].

Taken over from the parser port and still valid: elevation convention (QuaDRiGa names ESD, ESA, EoD;
`elevation = 90 deg - zenith`; the EoD offset has the opposite sign of the TR's ZOD offset); the formulas of the 27
v19 keys (3.0.4); the v19 conf files live in `data/config/` under Apache-2.0.

### 1.3 Out of scope

- Small-scale fading (SSF): cluster-number variability (7.6.15), near field (7.6.13), spatial non-stationarity
  (7.6.14), polarization variability (7.6.16), soft-LOS channel combination (7.6-19), clusters, rays, delays,
  angles. The keys exist already; section 8 lists what the SSF port needs from this one.
- Blockage (7.6.4), ISAC (7.9), map-based model (clause 8).
- LOS-state and indoor-state draws themselves (a `sos_rand` call in NG, D24), building-type draws, spatially
  consistent O2I variables (callers use `sos_rand` / `sos_randn` at the TR distances).
- Changes in QuaDRiGa-NG (section 10) and the merge into `main`.
- The generated documentation in `html_docu/` (rebuilt after the merge by `make documentation`).

### 1.4 Writer choices

Each is marked `[W-n]` where it applies. One line each; the reason follows in the text.

| # | Choice |
|---|---|
| W-1 | Floor separation in `lsf_draw` by per-floor phase offsets (Fourier offsets within blocks of `n_coeff` floor IDs, pseudo-random offsets from `qd_sos_rng(block)` between blocks), not by a coordinate offset. Fields of two floors are exactly uncorrelated at zero horizontal lag only (3.2.4). |
| W-2 | Dual mobility: the Tx side of a mirrored generator (ASD, ASA, ESD, ESA) is evaluated with the frequencies of its mirror partner (its phases are the partner's, slice 1), so swapping Tx and Rx swaps ASD/ASA and ESD/ESA exactly (bit for bit) also for unequal decorrelation distances and also with bias removal. |
| W-3 | Dual-mobility coupling of ASD/ASA and ESD/ESA uses the symmetric correlation `c = (A_g(d) + A_m(d)) / 2` instead of MATLAB's non-symmetric 2x2 matrix. |
| W-4 | Bias removal applies to generators 0 to 8 (Normal); the Uniform generators 9 (ground) and 10 (hE) are not corrected. |
| W-5 | Bias removal: every Normal generator is corrected once over the positions of its field (single mobility: Rx positions; dual mobility: the union of Rx and Tx positions), then slice 1 is refreshed by mirroring; one code path for both modes, mirror relation and swap reciprocity exact. (r0: per-device two-map correction; changed by R1-11.) |
| W-6 | Dual mobility with bias removal and drop positions but without `tx_pos` throws. |
| W-7 | 2D mode (`horizontal = true`): z is set to 0 for the SOS evaluation, the bias removal and the dual-mobility coupling distance. |
| W-8 | `path_loss` default `sf_sigma` is the SF standard deviation of the LSP formula (3.0.4), not `SF_sigma + SF_delta · lg(f)`, so that passing `sf_sigma` to `lsp_generate` is neutral for models without `sig` keys. |
| W-9 | Path-loss parameters without a default in `get_pl.m` are required; a missing one throws. Every `plpar` entry a model reads must have exactly one value. Breakpoint factors must be > 0. |
| W-10 | `satellite`: the argument of `asin` is limited to 1; a satellite at or below the horizon gives `PL = +Inf`. |
| W-11 | UMa random hE: `C'(hUT) = 1` above 23 m; an empty height set (hUT < 13.5 m) gives hE = 1 m; the mapping of one uniform value to the choice is fixed in 3.3.6. |
| W-12 | Path-loss constants in the v19 files are written with full double precision (e.g. `PL_E = 40/3` instead of 13.34), and RMa/SMa LOS use the TR's 2D breakpoint form (`PL_BP2D = 1`, W-41), so that the TR oracles hold to 1e-9 dB. |
| W-13 | `DS_VS` of InF LOS is 47/11 (average V/S of the two calibration halls), of InF NLOS DL/SH 50/11, written with full precision. |
| W-14 | The V/S term of the DS is active only if V/S > 0; `DS_VS > 0` with `DS_VS_a = DS_VS_b = 0` is a validation error; `vs_ratio > 0` with `DS_VS_a = DS_VS_b = 0` throws. |
| W-15 | All LOS probabilities are limited to [0, 1] (the TR UMa formula exceeds 1 slightly near 18 m for hUT > 13 m). |
| W-16 | InF-SH/DH and SMa outside the TR height ranges: fixed rules in 3.5.2. |
| W-17 | (r0: `36.873_UMa` with TR 36.873's `d^3`.) Dropped with the two 36.873 model names: both are identical to the 38.901 models (R1-12). |
| W-18 | 38.811 models: the tables of `set_scenario.m`, elevation from the absolute BS height `atan2(h_bs, d2d_out)`; no parameters. |
| W-19 | `o2i_loss`: `rnd` has 4 or 6 columns (fixed meaning); model names `low`, `high`, `low_A`, `high_v14`, `legacy_6GHz`, `car`, `car_metal`, `mmMAGIC_low`, `mmMAGIC_high`; the mmMAGIC elevation term is a separate output `loss_el` `[n_ut, n_tx]`. [pending user decision R1-10] |
| W-20 | Speed of light `c = 299792458` m/s wherever the library computes with c (2L/c, soft LOS wavelength); the TR's 3e8 in the breakpoints is part of the file constants. |
| W-21 | Inverse normal CDF for the soft LOS: Acklam's rational approximation with one Halley step on `std::erfc`, file-local in `src/los_probability.cpp`, tested through `los_probability`. |
| W-22 | `n_link = 0` (and `n_ut = 0`) is valid and gives empty outputs. |
| W-23 | Positions and all numeric array inputs must be finite; this is validated. |
| W-24 | Catch2 diagnostics use `tests/catch2_tests/qd_test_diag.hpp` with a new area `QD_DIAG_GBSM`, not the `QD_TEST_VERBOSE` define of SPEC_CPP_API (the repository moved to the area switch). |
| W-25 | MEX/Python: the path-loss model is the `model` entry of `plpar` (as `conf_file_read` returns it); outputs that need missing optional inputs are `[]` (MEX) or `None` (Python); `gr_eps` is complex in Python with the standard `complex` flag. |
| W-26 | The section description (`SECTION_DESC`) of "Geometry-based stochastic model" lives in `src/qd_scenpar.cpp`, `api_mex/conf_file_read.cpp` and `api_python/qpy_conf_file_read.cpp`, written in R4a (C++, MEX) and R4b (Python). |
| W-27 | Ground permittivity keeps QuaDRiGa's sign convention (imaginary part >= 0). |
| W-28 | `lsp_generate` uses the clamped `hBS` (>= 1e-4 m) in all terms, also in `ES_D_mu_C` and `ES_D_mu_D`. |
| W-29 | The functions complete and validate a copy of `scenpar` with `scenpar_validate(copy, "legacy", ..., validate = true)`: the value checks and the `DS_VS` cross-key rule of the key table apply to every key, also to keys the function does not use. (r0: per-function key checks; changed by R1-14.) |
| W-30 | Slice 0 of the `lsf_init_phases` output equals `sos_init_phases(n_coeff, 11, seed)` (documented contract). |
| W-31 | MEX and Python read `floor_id` as double and check that the values are non-negative integers. |
| W-32 | `winner_los` computes the breakpoint from the heights of each link (r0: mean over all links of the call; changed by R1-15). |
| W-33 | `LOS_state_lambda` in the 9 outdoor/indoor NLOS files, `IO_state_lambda` and `O2I_d2D_in_max` in all 8 O2I files. |
| W-34 | Writer sections of the 6 new keys as in 4.1. |
| W-35 | Legacy O2I distances with `d3D = 0`: indoor fraction 0. |
| W-36 | The random hE uses the d2D of the model (after the legacy O2I scaling, if any). |
| W-37 | `lsp_generate` validates column 9 of `rnd` (in [0, 1]) only when `gr_eps_re` or `gr_eps_im` is requested. |
| W-38 | The release-note text goes into the R4b report, not into `README.md` (see I-4). |
| W-39 | A shared internal header `src/quadriga_gbsm_helpers.hpp` (approved in the plan, 8.1) holds only helpers used by more than one source file (3.0.5); evaluation and bias removal stay file-local but are written generically over a generator table so that the SSF port can move them into the header. |
| W-40 | Missing `PL_hE_rand`, `PL_LOSmax`, `PL_Ax/Bx/Cx`, `PL_BP2D` keep today's behaviour (fixed hE, max with LOS, no extra term, 3D breakpoint form). |
| W-41 | `dual_slope` (and the LOS term of `nlos`) get the key `PL_BP2D`: non-zero evaluates the loss beyond the breakpoint as `PL1(dBP) + A2 · lg(d3D / dBP)` with the 2D breakpoint distance (TR RMa and SMa LOS); absent or 0 keeps the 3D form `PL1(dBP3) + A2 · lg(d3D / dBP3)` (TR UMa, UMi, InH, InF). Without it RMa/SMa differ from the TR by up to 0.034 dB (R1-02). |
| W-42 | `winner_los` uses `C2` beyond the breakpoint (`get_pl.m` uses `C1` in both branches, a bug: discontinuous at the breakpoint, 7.6 dB off at 3.5 GHz for WINNER C2 LOS) (R1-07). |
| W-43 | The bias removal treats all positions of `rx_pos` as one field (floor 0); for multi-floor drops the caller passes the positions of one floor or switches the bias removal off (R1-05). |
| W-44 | `lsf_draw` evaluates the sinusoids with `sos_val<double>` and `use_kernel = 0` (AVX2 kernel with single-precision sin/cos where available); the tests use the kernel tolerance for values produced by `lsf_draw` (5.0) (R1-01). |
| W-45 | `lsf_init_phases` and `lsf_draw` check the preset with `qd_sos_preset_lookup`; message `Input 'preset' is not a known preset.` (as `sos_rand`) (R1-28). |
| W-46 | `path_loss` has no `o2i_total_distance` flag: a given `d3d_in` selects the legacy outdoor-distance mode, its absence the TR geometric distance (R1-13). |
| W-47 | `los_probability` keeps the names `LOS`, `NLOS` (uniform group table in NG, 2 lines) and the 38.811 tables (QuaDRiGa NTN groups; about 20 lines either way). |

### 1.5 Issues with the plan

| # | Issue | Handling |
|---|---|---|
| I-1 | With bias removal in dual mobility (D21) and per-device corrections (r0), the Tx-side phases of ASD differ from the Rx-side phases of ASA by a constant, so the plan's test "swapping Tx and Rx swaps ASD and ASA" holds exactly only without bias removal. | Resolved in r1 (R1-11): each Normal field is corrected once over the union of Rx and Tx positions, then mirrored [W-5]; the swap holds exactly also with bias removal. |
| I-2 | `set_scenario.m` (36.873 UMa, line 367) uses `1.25e-6 · d^2 · exp(-d/150)`; TR 36.873 has `d^3` (same as 38.901's `5/4 · (d/100)^3`). | Resolved in r1 (R1-12): with `d^3` the 36.873 UMa and UMi models equal the 38.901 ones; both names and `indoor_ut` are dropped, NG maps its 3GPP_3D groups to `38.901_UMa` / `38.901_UMi`; the `d^2` is a bug and gets no legacy flag (LOS probabilities of indoor UTs above 13 m change in NG as a bug fix). |
| I-3 | Plan 6.9: "`DS_VS` = 4.27 reproduces today's value" of InF LOS (`DS_mu = -7.2535`). It does not: -7.2535 is the mean of the two calibration-hall values of lgDS; V/S = 4.27 gives -7.2530, 47/11 gives -7.2528. | Spec uses 47/11 [W-13]; the change of the InF LOS DS median is 0.07 % in lg. |
| I-4 | WP-7 asks for release notes in `README.md`; README has no release-notes section. | R4b writes the text into its report [W-38]; Stephan decides where it goes. |
| I-5 | The v19 files round derived path-loss constants (`PL_E = 13.34` instead of 40/3, RMa `PL_D = 0.0014` instead of 0.0013979, ...). TR oracles typed into the tests would need tolerances of about 0.03 dB, which hide real errors. | R3 writes the constants with full precision [W-12] and switches RMa/SMa LOS to the TR's 2D breakpoint form [W-41]; the path loss of these files changes by up to 0.02 dB (constants) and up to 0.034 dB (breakpoint form, RMa/SMa beyond dBP). |
| I-6 | `gen_o2i_loss.m` draws the O2I deviation with `qd_sos.rand` (uniform) times sigma_P, not with a Normal variable. | The library uses N(0,1) (TR). No parity of the O2I deviation; it is not part of D23. |
| I-7 | `get_pl.m` model `nlos` adds the O2I loss twice when the LOS term wins (once in the recursive `dual_slope` call, once at the end). | The library never adds O2I loss in `path_loss` (plan 6.5); NG must not replicate the quirk. Noted in section 10. |
| I-8 | Plan 6.8 lists one model `mmMAGIC`; `gen_o2i_loss.m` mixes the mmMAGIC indoor and elevation terms with the low-loss and the high-loss wall of TR v14 (IRR glass 23 + 0.3 f). One name cannot carry the wall type when models are mixed by weights. | Names `mmMAGIC_low`, `mmMAGIC_high`, plus `high_v14` for the legacy 3GPP high-loss wall [W-19]. Model names were left to review point A by the plan (section 7). [pending user decision R1-10: moot if the O2I scope is reduced] |

### 1.6 Differences to the SPEC files (approved)

- D12: `double` only, no templates, no explicit instantiations, no float/double parity tests.
- D5: degrees and linear power ratios in outputs.
- Argument order of the plan's signatures (outputs before optional inputs) is the approved contract.
- [W-24]: diagnostics via `qd_test_diag.hpp`, area `QD_DIAG_GBSM`.
- Review points are not stops (see header).
- New internal header `src/quadriga_gbsm_helpers.hpp` (approved, plan 8.1), limited to helpers used by more than
  one source file [W-39].
- Each implementation run adds to `include/quadriga_gbsm.hpp` only the declarations it defines (section 7), so
  that no intermediate state declares functions without definition.


## 2. Public API

### 2.1 Header `include/quadriga_gbsm.hpp` (final text)

The final content of the file is given below. R1 creates it with the include guard, the includes, the
`static_assert` lines, the "Scenario parameters" group and the comment block of the "Large-scale fading" group
(generator index); the four moved declarations are copied unchanged, including their comments, from
`include/quadriga_tools.hpp` lines 517 to 565, and removed there together with the group line
`// ---- Scenario parameters ----`. Each later run adds the declarations it defines, at their place in this text
[R1-27]: R2a `lsf_init_phases`, `lsf_draw`; R2b `lsp_generate`; R3a `path_loss`; R3b `los_probability`,
`o2i_loss`. `include/quadriga_lib.hpp` includes the header after `quadriga_sos.hpp`.

```cpp
// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#ifndef quadriga_gbsm_H
#define quadriga_gbsm_H

#include <armadillo>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

// If arma::uword and size_t are not the same width (e.g. 64 bit), the compiler will throw an error here
// This allows the use of "uword", "size_t" and "unsigned long long" interchangeably
// This requires a 64 bit platform, but will compile on Linux, Windows and macOS
static_assert(sizeof(arma::uword) == sizeof(unsigned long long), "arma::uword and unsigned long long have different sizes");
static_assert(sizeof(size_t) == sizeof(unsigned long long), "size_t and unsigned long long have different sizes");

namespace quadriga_lib
{

    // ---- Scenario parameters ----

    // (conf_file_read, scenpar_validate, conf_file_list, conf_file_write: moved unchanged from quadriga_tools.hpp)

    // ---- Large-scale fading ----

    // Generators of the large-scale fading (LSF) phase set, column index of 'phase' and 'rnd':
    //   0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR      Normal(0,1), decorrelation distance of the LSP
    //   8 absolute TOA                                           Normal(0,1), absTOA_lambda
    //   9 ground permittivity                                    Uniform(0,1), SC_lambda
    //  10 UMa environment height hE                              Uniform(0,1), SF_lambda
    // - Slice 1 (Tx side) of 'phase' is slice 0 (Rx side) of the mirror generator: ASD <-> ASA, ESD <-> ESA, all
    //   others mirror themselves; a decorrelation distance of 0 is used as 0.1 m
    // - The functions are documented in src/lsf_init_phases.cpp, src/lsf_draw.cpp, src/path_loss.cpp,
    //   src/lsp_generate.cpp, src/los_probability.cpp, src/o2i_loss.cpp

    // Draw the random phases of the large-scale generators of one BS site and scenario
    // - Generator order and reciprocity rules: see the LSF generator index above
    // - Bias removal over the positions in 'rx_pos' (and 'tx_pos' for dual mobility) when given and n_link > 2
    void lsf_init_phases(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances)
                         arma::cube &phase,                                      // Output: Phases in [rad], [n_coeff, 11, 2]; slice 0 = Rx side, slice 1 = Tx side
                         arma::sword seed = -1,                                  // Seed, -1 = system random device
                         const arma::mat *rx_pos = nullptr,                      // Optional: Rx positions of the drop (one floor) for the bias removal, [n_link, 3]
                         const arma::mat *tx_pos = nullptr,                      // Optional: Tx positions, [1, 3] or [n_link, 3]
                         bool dual_mobility = false,                             // Tx and Rx both move: the Tx side is a second device
                         bool remove_bias = true,                                // Remove the sample mean over the positions
                         const std::string &preset = "Comb256",                  // SOS preset, see sos_preset
                         bool horizontal = true);                                // true: 2D correlation in the x-y plane (TR); false: 3D (legacy)

    // Spatially correlated normalized random values of the large-scale generators at a set of links
    // - Columns 0 to 8 are Normal(0,1), columns 9 and 10 Uniform(0,1)
    // - Dual mobility: ASD/ASA and ESD/ESA are coupled by the ACF at the Tx-Rx distance
    // - Links with different floor IDs are uncorrelated at equal horizontal positions; equal IDs share one field
    void lsf_draw(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances)
                  const arma::cube &phase,                                 // Phases from lsf_init_phases, [n_coeff, 11, 2]
                  const arma::mat &tx_pos,                                 // Tx positions, [1, 3] or [n_link, 3]
                  const arma::mat &rx_pos,                                 // Rx positions, [n_link, 3]
                  arma::mat &rnd,                                          // Output: Random values, [n_link, 11]
                  const arma::uvec *floor_id = nullptr,                    // Optional: Floor index per link, [n_link]; nullptr = all on one floor
                  bool dual_mobility = false,                              // As in lsf_init_phases
                  const std::string &preset = "Comb256",                   // Must match lsf_init_phases
                  bool horizontal = true);                                 // Must match lsf_init_phases

    // Path loss and shadow-fading standard deviation of a set of links
    // - Models: logdist, logdist_simple, constant, winner_los, winner_nlos, winner_pathloss, dual_slope,
    //   tripple_slope, nlos, satellite; parameters in 'plpar' as in the conf files (keys without "PL_")
    // - An empty 'pl_model' gives zero loss
    void path_loss(const std::string &pl_model,                                          // Path-loss model, value of 'PL_model'
                   const std::unordered_map<std::string, std::vector<double>> &plpar,    // Path-loss parameters
                   const std::unordered_map<std::string, double> &scenpar,               // Scenario parameters (SF std, LSP_fc_min)
                   const arma::vec &fc,                                                  // Carrier frequencies in [Hz], [n_freq]
                   const arma::mat &tx_pos,                                              // Tx positions, [1, 3] or [n_link, 3]
                   const arma::mat &rx_pos,                                              // Rx positions, [n_link, 3]
                   arma::mat &pl,                                                        // Output: Path loss in [dB], [n_link, n_freq]
                   arma::mat *sf_sigma = nullptr,                                        // Optional output: SF standard deviation in [dB], [n_link, n_freq]
                   const arma::vec *d3d_in = nullptr,                                    // Optional: Indoor part of the 3D distance in [m], [n_link]; given = outdoor distance only (legacy)
                   const arma::vec *rnd_he = nullptr);                                   // Optional: Uniform(0,1) per link for the random hE, column 10 of lsf_draw, [n_link]

    // Large-scale parameters of a set of links from normalized random values
    // - Mean and std per link and frequency from the scenario parameters (incl. LSP_fc_min, ES_D_mu_B/C/D);
    //   cross-correlation by the root of the LSP cross-correlation matrix; transformation to linear values
    // - Same random values for all frequencies (TR 7.6.5)
    void lsp_generate(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters
                      const arma::vec &fc,                                    // Carrier frequencies in [Hz], [n_freq]
                      const arma::mat &tx_pos,                                // Tx positions, [1, 3] or [n_link, 3]
                      const arma::mat &rx_pos,                                // Rx positions, [n_link, 3]
                      const arma::mat &rnd,                                   // Random values from lsf_draw, [n_link, 11]
                      arma::cube &lsp,                                        // Output: [n_link, n_freq, 8]: DS [s], KF, SF [linear], ASD, ASA, ESD, ESA [deg], XPR [linear]
                      const arma::mat *sf_sigma = nullptr,                    // Optional: SF std in [dB] from path_loss, replaces the scenpar value, [n_link, n_freq]
                      arma::vec *abs_toa = nullptr,                           // Optional output: Absolute TOA offset in [s], 0 if disabled, [n_link]
                      arma::mat *gr_eps_re = nullptr,                         // Optional output: Ground permittivity, real part, [n_link, n_freq]
                      arma::mat *gr_eps_im = nullptr,                         // Optional output: Ground permittivity, imaginary part, [n_link, n_freq]
                      arma::mat *eod_offset_deg = nullptr,                    // Optional output: EoD offset in [deg], positive = up, [n_link, n_freq]
                      arma::cube *lsp_mu = nullptr,                           // Optional output: Mean in the log domain (lg or dB), [n_link, n_freq, 8]
                      arma::cube *lsp_sigma = nullptr,                        // Optional output: Std in the log domain, [n_link, n_freq, 8]
                      double vs_ratio = 0.0,                                  // InF hall volume / surface in [m], overrides the key DS_VS; 0 = use DS_VS
                      bool cholesky = true,                                   // true: Cholesky in TR order; false: sqrtm (legacy)
                      bool clip_spreads = true);                              // Clip AS to 104 deg, ES to 52 deg (TR step 4)

    // LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models
    // - Models: "38.901_RMa", "38.901_UMi", "38.901_UMa", "38.901_InH_Mixed", "38.901_InH_Open",
    //   "38.901_InF_SL", "38.901_InF_DL", "38.901_InF_SH", "38.901_InF_DH", "38.901_InF_HH", "38.901_SMa",
    //   "38.811_DenseUrban", "38.811_Urban", "38.811_Suburban", "38.811_Rural", "QuaDRiGa_Industrial",
    //   "LOS" (always 1), "NLOS" (always 0)
    // - Soft LOS (7.6-18) when 'rnd_g', 'fc' and 'los_soft' are given
    void los_probability(const std::string &model,          // Model name
                         const arma::vec &d2d_out,          // 2D outdoor distance in [m] (indoor scenarios: 2D distance), [n_link]
                         const arma::vec &h_bs,             // BS height in [m], [n_link] or [1]
                         const arma::vec &h_ut,             // UT height in [m], [n_link] or [1]
                         arma::vec &p_los,                  // Output: LOS probability, [n_link]
                         double d_clutter = NAN,            // InF, SMa: clutter size in [m]; NaN = TR default of the model
                         double r_clutter = NAN,            // InF: clutter density; SMa: vegetation density r_vegetation; NaN = default
                         double h_clutter = NAN,            // InF SH, DH: clutter height in [m]; NaN = default
                         const arma::vec *rnd_g = nullptr,  // Optional: Normal(0,1) per link for the soft LOS state, [n_link]
                         const arma::vec *fc = nullptr,     // Optional: Carrier frequencies in [Hz] for the soft LOS state, [n_freq]
                         arma::mat *los_soft = nullptr);    // Optional output: Soft LOS state in [0, 1], [n_link, n_freq]

    // O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3
    // - Models: "low", "high", "low_A" (Table 7.4.3-2), "high_v14" (TR v14 IRR glass), "legacy_6GHz" (Table 7.4.3-3),
    //   "car", "car_metal" (7.4.3.2), "mmMAGIC_low", "mmMAGIC_high"
    // - The model of each UT is chosen by comparing column 0 of 'rnd' with the cumulative 'weights'
    void o2i_loss(const arma::vec &fc,                       // Carrier frequencies in [Hz], [n_freq]
                  const arma::mat &rnd,                      // Random values per UT, [n_ut, 4] or [n_ut, 6]: U model, U d2D-in 1, U d2D-in 2, N sigma_P, U alpha, U d_room
                  arma::mat &loss,                           // Output: Penetration loss in [dB] without the mmMAGIC elevation term, [n_ut, n_freq]
                  const std::vector<std::string> &models,    // Models to mix, [n_model]
                  const arma::vec &weights,                  // Probability of each model, sum 1, [n_model]
                  double d2d_in_max = 25.0,                  // Maximum indoor 2D distance in [m] (key O2I_d2D_in_max)
                  arma::vec *d2d_in = nullptr,               // Optional output: Indoor 2D distance in [m], 0 for car models, [n_ut]
                  arma::uvec *model_index = nullptr,         // Optional output: Chosen model per UT, 0-based index into 'models', [n_ut]
                  const arma::mat *tx_pos = nullptr,         // Optional: Tx positions for d3d_in and loss_el, [n_tx, 3]
                  const arma::mat *rx_pos = nullptr,         // Optional: UT positions, [n_ut, 3]
                  arma::mat *d3d_in = nullptr,               // Optional output: Indoor part of the 3D distance in [m], [n_ut, n_tx]
                  arma::mat *loss_el = nullptr,              // Optional output: mmMAGIC elevation loss in [dB], 0 for other models, [n_ut, n_tx]
                  bool d2d_in_min_of_two = true);            // true: min of two uniforms (TR); false: legacy inverse CDF of one uniform
}

#endif
```

The four moved declarations are inserted verbatim at the comment "(conf_file_read, ...: moved unchanged ...)",
replacing that comment line. Changes against the plan's section 6: `o2i_loss` gets `loss_el` and the model names
of [W-19] (review point A of the plan); `path_loss` has no `o2i_total_distance` (a given `d3d_in` selects the
legacy mode [W-46]); `los_probability` has no `indoor_ut` and no 36.873 models (R1-12); all other declarations are
the plan's in names, types, order and defaults (comments are refined).

Pending decisions with a local edit in this header:
- [pending user decision R1-09] Dropping the legacy root removes the argument `bool cholesky = true` of
  `lsp_generate` (and its comment); `clip_spreads` then follows `lsp_sigma`.
- [pending user decision R1-10] Reducing the O2I scope gives `o2i_loss` the models "low", "high", "low_A",
  "legacy_6GHz", "car", "car_metal", `rnd` `[n_ut, 4]` (U model, U d2D-in 1, U d2D-in 2, N sigma_P), and removes
  `tx_pos`, `rx_pos`, `d3d_in`, `loss_el`, `d2d_in_min_of_two`; `path_loss` then takes
  `const arma::vec *d2d_in = nullptr` (indoor 2D distance, legacy mode) instead of `d3d_in` (3.6.0).

### 2.2 LSF generator index

| g | Generator | Distribution | Decorrelation distance λ_g | Mirror m(g) | Bias removal |
|---|---|---|---|---|---|
| 0 | DS | Normal | `DS_lambda` | 0 | yes |
| 1 | KF | Normal | `KF_lambda` | 1 | yes |
| 2 | SF | Normal | `SF_lambda` | 2 | yes |
| 3 | ASD | Normal | `AS_D_lambda` | 4 | yes |
| 4 | ASA | Normal | `AS_A_lambda` | 3 | yes |
| 5 | ESD | Normal | `ES_D_lambda` | 6 | yes |
| 6 | ESA | Normal | `ES_A_lambda` | 5 | yes |
| 7 | XPR | Normal | `XPR_lambda` | 7 | yes |
| 8 | absolute TOA | Normal | `absTOA_lambda` | 8 | yes |
| 9 | ground permittivity | Uniform | `SC_lambda` | 9 | no [W-4] |
| 10 | UMa environment height hE | Uniform | `SF_lambda` [D19] | 10 | no [W-4] |

A decorrelation distance of 0 is used as 0.1 m (as `sos_randn`). The generator of column 8 exists also when the
absolute TOA is disabled (`absTOA_mu <= -30`); `lsp_generate` then ignores it.

### 2.3 Resolved review-point-A items of the plan (section 7 of the plan)

| Item | Resolution |
|---|---|
| `rnd` columns of `o2i_loss` | 4 or 6 columns: 0 U model choice, 1 U d2D-in #1, 2 U d2D-in #2, 3 N(0,1) deviation (sigma_P, car), 4 U mmMAGIC alpha, 5 U mmMAGIC d_room. 6 columns are required when `models` contains a mmMAGIC model, otherwise 4 or 6. [W-19] [pending user decision R1-10: 4 columns only] |
| mmMAGIC output shape | `loss` stays `[n_ut, n_freq]` without the elevation term; the elevation term is the new optional output `loss_el` `[n_ut, n_tx]`, frequency-independent. The caller adds `loss_el(:, t)` for BS t. [W-19] [pending user decision R1-10: no `loss_el`] |
| mmMAGIC model names | `mmMAGIC_low`, `mmMAGIC_high` (wall loss low / high of TR v14) instead of `mmMAGIC` [I-8]. [pending user decision R1-10] |
| 38.811 table handling | Fixed tables inside `los_probability` (Appendix 11.2), linear interpolation over the elevation, no parameters; `d_clutter`, `r_clutter`, `h_clutter` are ignored by these models. [W-18] |
| Floor separation method | Per-floor phase offsets, not a coordinate offset (3.2.4) [W-1]. Reason: works in 2D and 3D mode, keeps the mirror relations, needs no large coordinate offsets (which would push small decorrelation distances towards the precision limit of the kernels), and gives exactly zero correlation at zero horizontal lag (UTs stacked above each other, the most relevant case) between floors whose IDs differ by less than `n_coeff`. At other lags the cross-covariance of two floors is not zero: up to 0.11 to 0.16 for the 256-sinusoid presets and 0.07 to 0.10 for the 1024-sinusoid presets (2D mode, lags 0 to 60 m), the same level as for pseudo-random offsets (R1-16). |


## 3. Normative behaviour

### 3.0 Shared definitions

#### 3.0.1 Notation and conventions

- `lg(x) = log10(x)`, `ln` the natural logarithm. Angles in formulas are in radians unless the name ends in `_deg`
  or the text says degrees. `atan` is the one-argument arctangent, `atan2(y, x)` the two-argument one.
- `f = fc / 1e9` is the carrier frequency in GHz; `fc` is always given in Hz. `n_freq = fc.n_elem`.
- `c = 299792458` m/s wherever the library computes with the speed of light [W-20]. The TR breakpoint formulas use
  `c = 3e8`; that value is part of the conf-file constants (`PL_E = 40/3`, `20·pi/3`), not of the code.
- Positions are rows `{x, y, z}` in meters. `n_link = rx_pos.n_rows`. `tx_pos` has 1 row (used for every link) or
  `n_link` rows; `tx(l)` denotes row 0 or row `l`.
- Link geometry: `dx = rx(l) - tx(l)`; `d2D = sqrt(dx_x² + dx_y²)`; `d3D = sqrt(d2D² + dx_z²)`; `hBS = tx(l).z`;
  `hUT = rx(l).z`. All heights are z coordinates (no ground model).
- `n_link = 0` is valid in all new functions: `rx_pos` is `[0, 3]`, outputs get 0 rows [W-22]. `tx_pos` may then
  have 0 or 1 rows.
- Optional outputs (`nullptr`) are neither allocated nor computed. Outputs are resized only if their size differs,
  then every element is written (SPEC_CPP_API "Outputs"). All validation runs before any output is touched.
- Exceptions: `std::invalid_argument` for invalid input; messages follow SPEC_CPP_API (full sentence, argument in
  single quotes). Exceptions thrown by called library functions (`scenpar_validate`, ...) pass through unchanged.
- Precision [W-44]: all functions compute in double. The only single-precision part is the sine/cosine evaluation
  inside `sos_val`, which `lsf_draw` calls as `sos_val<double>` with `use_kernel = 0`: on CPUs with AVX2 the kernel
  reduces the argument in double, evaluates sin/cos in single precision and accumulates in double, with a
  documented deviation below `4e-7 · sqrt(2 · n_coeff)` (9e-6 for 256, 1.8e-5 for 1024 sinusoids). The float
  instantiation of `sos_val` is not used (it would only convert inputs and outputs). `sos_remove_bias` (closed
  form), `lsp_generate`, `path_loss`, `los_probability` and `o2i_loss` are pure double. Downstream LSPs inherit a
  relative error of about `ln(10) · sigma · 1e-5`, which is negligible. Test tolerances: 5.0.

#### 3.0.2 Common validation (in this order, before the function-specific checks)

| Check | Message |
|---|---|
| `rx_pos.n_cols != 3` | `Input 'rx_pos' must have 3 columns.` |
| `tx_pos.n_cols != 3` | `Input 'tx_pos' must have 3 columns.` |
| `tx_pos.n_rows` not 1 and not `n_link` (0 allowed if `n_link = 0`) | `Number of rows in 'tx_pos' must be 1 or match the number of rows in 'rx_pos'.` |
| a non-finite entry in `rx_pos` / `tx_pos` [W-23] | `Input 'rx_pos' must be finite.` / `Input 'tx_pos' must be finite.` |
| `fc` empty (functions that take `fc`) | `Input 'fc' cannot be empty.` |
| an entry of `fc` not finite or `<= 0` | `Input 'fc' must be finite and > 0.` |
| optional vector input with wrong length | `Number of elements in '<name>' does not match number of rows in 'rx_pos'.` |

#### 3.0.3 Scenario parameters inside the functions

Each function that takes `scenpar` completes and validates a copy with `scenpar_validate(copy, "legacy",
&lsp_xcorr_or_nullptr, nullptr, true)` [W-29]: missing keys get their default [D9]; an unknown key throws
(`Unknown key 'X' in 'scenpar'.`); every value is checked by its validation class (e.g.
`Value of 'XPR_lambda' must be finite and >= 0.`), and the cross-key rules of the key table apply (incl. the
`DS_VS` rule of 4.2). These messages pass through unchanged; the functions have no own key checks. Consequence
(accepted): an invalid value of a key that the function does not use throws as well; a `scenpar` from
`conf_file_read` or `scenpar_validate` never contains one. The return value of `scenpar_validate` (positive
definite) is used by `lsp_generate` only.

#### 3.0.4 LSP mean and standard deviation (shared by `lsp_generate` and the default `sf_sigma` of `path_loss`)

Source: `qd_builder.m` `get.lsp_vals` (lines 233 to 309), `gen_lsf_parameters.m` lines 232 to 268, v19 key formulas of the parser plan (plan 5.1). LSP index `i = 0..7` with key prefix
`P_i = DS, KF, SF, AS_D, AS_A, ES_D, ES_A, XPR`.

Per link `l` and frequency `k`:

```
d   = max(d2D, 1e-4)                    clamped 2D distance [m]
h   = max(hBS, 1e-4)                    clamped BS height [m]           [W-28: used in all terms]
a   = atan(h / d)                       elevation of the BS seen from the UT [rad]
f'  = max(f, LSP_fc_min)                frequency for all LSP terms [GHz]; LSP_fc_min = 0 gives f' = f

mu_i    = P_mu + P_gamma · lg(P_omega + f') + P_epsilon · lg(d) + P_zeta · lg(h) + P_alpha · lg(a)
sigma_i = max(0, P_sigma + P_delta · lg(P_omega + f')) + P_kappa · lg(d) + P_tau · lg(h) + P_beta · lg(a)
```

- SF (`i = 2`) has no keys `SF_mu`, `SF_gamma`, `SF_epsilon`, `SF_zeta`, `SF_alpha`: they are 0, so `mu_2 = 0`.
  `SF_omega`, `SF_sigma`, `SF_delta`, `SF_kappa`, `SF_tau`, `SF_beta` exist.
- DS (`i = 0`), V/S term [D26, W-14]: with `VS = vs_ratio` if `vs_ratio > 0`, else `VS = DS_VS`:
  if `VS > 0`: `mu_0 += lg(DS_VS_a · VS + DS_VS_b)`.
- ESD (`i = 5`), v19 terms, applied after the general formula and before the floor:
  `mu_5 += ES_D_mu_A · d / 1000 + ES_D_mu_B · hUT + ES_D_mu_C · abs(hUT - h) + ES_D_mu_D · max(hUT - h, 0)`, then
  `mu_5 = max(mu_5, ES_D_mu_min)` (`ES_D_mu_min` defaults to `-Inf`). `hUT` is the raw z of the Rx.
- Units: DS mean and std in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF, XPR in dB.
- The same `(l, k)` values serve `lsp_mu`, `lsp_sigma` and the transformation in 3.4.

#### 3.0.5 Internal header `src/quadriga_gbsm_helpers.hpp` [W-39]

Created in R2a, extended in R2b; included by the source files named below and by nothing public. It holds only
`static inline` helpers used by more than one source file; names start with `qd_gbsm_`. The header starts with a
documentation block (comment lines) that states: purpose (helpers shared by the source files of the
geometry-based stochastic model, not part of the public API), the users of each helper, the LSF generator index
(table of 2.2), the slice convention of the phase cube (slice 0 Rx side, slice 1 Tx side = mirror of slice 0),
and the units (positions in m, frequencies in Hz, decorrelation distances in m). Required content (signatures may
be adapted by the implementer; the report states the final form):

```cpp
// One generator of a phase set: decorrelation distance in [m] (0 already replaced by 0.1 m), distribution,
// mirror partner, bias removal on/off
struct qd_gbsm_gen { double lambda; bool uniform; arma::uword mirror; bool bias; };

// The 11 LSF generators (index table) from a completed scenpar             users: lsf_init_phases.cpp, lsf_draw.cpp
static inline std::vector<qd_gbsm_gen> qd_gbsm_lsf_generators(const std::unordered_map<std::string, double> &sp);

// Frequencies of every generator, sos_preset(preset, freq[g], &acf[g], &dist[g], gen[g].lambda); throws for an
// unknown preset with "Input 'preset' is not a known preset." (checked with qd_sos_preset_lookup); returns n_coeff
//                                                                          users: lsf_init_phases.cpp, lsf_draw.cpp
static inline arma::uword qd_gbsm_freq(const std::string &preset, const std::vector<qd_gbsm_gen> &gen,
                                       std::vector<arma::mat> &freq, std::vector<arma::vec> &acf,
                                       std::vector<arma::vec> &dist);

// Positions for the evaluation: R = rx_pos, T = tx_pos broadcast to n_link rows; z = 0 in 2D mode
//                                                                          users: lsf_init_phases.cpp, lsf_draw.cpp
static inline void qd_gbsm_positions(const arma::mat &rx_pos, const arma::mat &tx_pos, bool horizontal,
                                     arma::mat &R, arma::mat &T);

// LSP mean and std of all links and frequencies, [n_link, n_freq, 8] each   users: lsp_generate.cpp, path_loss.cpp
static inline void qd_gbsm_lsp_mu_sigma(const std::unordered_map<std::string, double> &sp, const arma::vec &fc,
                                        const arma::mat &tx_pos, const arma::mat &rx_pos, double vs_ratio,
                                        arma::cube &mu, arma::cube &sigma);

// Cholesky-Banachiewicz of a symmetric 8x8 matrix, column-major double[64]: A = L·L^T, L lower triangular with
// zeros above the diagonal; returns false at the first pivot that is not > 0 (NaN included)
//                                                                          users: lsp_generate.cpp, qd_scenpar.cpp
static inline bool qd_gbsm_chol8(const double *A, double *L);
```

`qd_scenpar_xcorr` in `src/qd_scenpar.cpp` is refactored in R2b to call `qd_gbsm_chol8` (its own Cholesky loop is
removed; behaviour unchanged, the existing conf tests cover it), so that the library has one 8x8
Cholesky.

File-local (anonymous namespace or lambda in the one file that uses them): the generator evaluation with floors,
dual mobility and coupling (`lsf_draw.cpp`), the bias removal (`lsf_init_phases.cpp`), the floor offsets
(`lsf_draw.cpp`), the legacy symmetric root (`lsp_generate.cpp`, [pending user decision R1-09]), the probit
(`los_probability.cpp`), the path-loss models, the tables of section 11. The evaluation and the bias removal take
the generator table (`std::vector<qd_gbsm_gen>`), the frequency tables and the phase cube as inputs and contain no
LSF-specific constants, so that the later SSF port can move them into the header unchanged and evaluate its own
phase sets (cluster delays, angles, XPR, phases) with the same rules for mirroring, floors, 2D/3D and dual
mobility (section 8). A generator `g` with `mirror != g` is a pair member; the coupling of 3.2 step 5 is applied
to every pair `(g, m)` with `g < m` whose generators are Normal.

### 3.1 `lsf_init_phases`

Source: `init_sos.m` lines 112 to 160 (LSF part), `qd_sos.m` `init`, `remove_bias`; decisions D3, D13, D14, D21.

**Validation** (after 3.0.2 for the inputs that are given; `rx_pos` and `tx_pos` are optional here):

1. `preset` known (`qd_sos_preset_lookup`), else `Input 'preset' is not a known preset.` [W-45]; `n_coeff` from the
   lookup.
2. scenpar completed and validated (3.0.3); this covers the 10 decorrelation-distance keys of 2.2.
3. `rx_pos` (if given): 3 columns, finite. `tx_pos` (if given): 3 columns, finite; if `rx_pos` is given as well,
   1 or `n_link` rows.
4. `dual_mobility && remove_bias && rx_pos != nullptr && n_link > 2 && tx_pos == nullptr`:
   `Input 'tx_pos' is required for the bias removal with dual mobility.` [W-6]

**Algorithm**

1. `sp` = completed scenpar; generator table of 2.2 with `lambda'_g = lambda_g > 0 ? lambda_g : 0.1`.
2. `P0 = sos_init_phases(n_coeff, 11, seed)`, `[n_coeff, 11]` (slice 0, Rx side) [W-30].
3. Bias removal, only if `remove_bias && rx_pos != nullptr && n_link > 2` (`n_link = rx_pos->n_rows`; with fewer
   positions, including 0, no SOS function is called) [W-5]:
   - Positions `R`, `T` from `qd_gbsm_positions` (Tx broadcast to `n_link` rows, z = 0 in 2D mode [W-7]).
   - Positions of the field: `U = R` in single mobility (`tx_pos` is not used), `U = [R; T]` (`2·n_link` rows,
     Rx rows first) in dual mobility.
   - For every Normal generator g = 0..8: `sos_remove_bias(F_g, P0.col(g), U)` (one device; `F_g` from
     `qd_gbsm_freq`). `P0.col(g)` stands for a `[n_coeff, 1]` matrix copied out and written back. Generators that
     share `lambda'` may be corrected in one call (same result: the offset is computed per column).
   - Generators 9 and 10 are never corrected [W-4].
4. Mirror: `P1(:, g) = P0(:, m(g))` for all g (slice 1, Tx side) [D21: the Tx side is the mirror of the corrected
   Rx side, so the mirror relation holds exactly with and without bias removal].
5. `phase` = `[n_coeff, 11, 2]`, slice 0 = `P0`, slice 1 = `P1`.

Why one correction over the union in dual mobility [W-5]: the field of a generator is shared by both devices
(the Tx side of ASD is the ASA field), so its sample mean is removed over all positions where it is evaluated.
Mirror relations and the swap reciprocity of `lsf_draw` then hold exactly, and single and dual mobility share one
code path. Swapping Tx and Rx positions gives the same union and therefore the same phases.

**Properties to document**: same seed gives the same phases; without bias removal slice 0 equals
`sos_init_phases(n_coeff, 11, seed)`; slice 1 is always the mirror of slice 0; the sample mean of every Normal field
over the positions is 0 (single mobility: over `rx_pos`; dual mobility: over the Rx and Tx positions together);
consequently the link mean of the `lsf_draw` values is 0 in single mobility, and in dual mobility for DS, KF, SF,
XPR and TOA when all links have the same Tx-Rx distance (the normalization `1/sqrt(2 + 2·A(D))` is then the same
for all links), while the ASD/ASA and ESD/ESA columns have a zero link mean only approximately (their two devices
see different fields, and the coupling mixes them); bias removal over several floors [W-43]: all positions in
`rx_pos` are treated as one field (floor 0); for a drop with several floors pass the positions of one floor (e.g.
the links of floor 0) or set `remove_bias = false`, the fields of other floors keep their sample mean; `tx_pos` is
ignored without dual mobility; uncorrected generators 9, 10.

Difference to QuaDRiGa (stated in section 10 and the release notes; not in the documentation block, which describes
only the library's rule): `init_sos.m` removes the bias before the mirroring, and the mirroring overwrites the
Rx-side phases of ASD with the Tx-side phases of ASA (and vice versa; the same for ESD/ESA). The correction these
phases carry was computed with the partner's frequencies, so it holds only when the two decorrelation distances
are equal. This affects single mobility as well: whenever ASD/ASA or ESD/ESA have different decorrelation
distances (e.g. UMa LOS: ASD 18 m, ASA 15 m), MATLAB's ASD/ASA and ESD/ESA fields keep a non-zero drop mean, while
the library removes it; the bias-removed values change against today in single mobility too, and in dual mobility
in general (D21) (R1-17).

**Output**: `phase` `[n_coeff, 11, 2]`, phases in rad: uncorrected generators in [-pi, pi) (`sos_init_phases`),
corrected generators in [-pi, pi) as returned by `sos_remove_bias`.

### 3.2 `lsf_draw`

Source: `gen_lsf_parameters.m` lines 192 to 226 (`val` of the 8 generators, dual-mobility coupling), `qd_sos.m` `val`/`eval_mode`/`sos_eval`; decisions D6, D13, D15, D19, D20.

**Validation**: 3.0.2 for `tx_pos`, `rx_pos`; `preset` known (`Input 'preset' is not a known preset.` [W-45]);
scenpar completed and validated (3.0.3); `phase` must be `[n_coeff, 11, 2]` with `n_coeff` of the preset:
`Input 'phase' must have size [<n_coeff>, 11, 2] for preset '<preset>'.`; `phase` finite:
`Input 'phase' must be finite.`; `floor_id` (if given) `n_link` elements.

**Algorithm**

0. `n_link = 0`: `rnd` is resized to `[0, 11]` and the function returns after the validation; no SOS function is
   called (they throw on empty positions) [W-22].
1. Positions `R`, `T` from `qd_gbsm_positions` (Tx broadcast to `n_link` rows; z = 0 in 2D mode [W-7]). In
   single mobility `T` is not used (the values do not depend on `tx_pos`).
2. Generator table of 2.2 and `F_g, acf_g, dist_g` from `qd_gbsm_freq` (`lambda'_g`); `uniform_g` = true for g = 9,
   10.
3. Floors: `fl(l) = floor_id ? (*floor_id)(l) : 0`. Links are grouped by distinct floor ID. For floor `q` the phases
   are `P0_q = P0 + Delta_q`, `P1_q = P1 + Delta_q` (the same offset vector added to every column of both slices),
   `Delta_q` from 3.2.4. Floor 0 has `Delta_0 = 0`, so `floor_id` all 0 gives exactly the values of
   `floor_id = nullptr`.
4. Fields (all generators evaluated as Normal, `uniform = false`), for every floor group (links `L`) and generator
   g, with `sos_val<double>` in one-device mode and `use_kernel = 0` [W-44]:
   - Rx side: `a_g = sos_val(F_g, P0_q.col(g), R.rows(L))`.
   - Tx side (dual mobility only): `b_g = sos_val(F_m(g), P1_q.col(g), T.rows(L))`, i.e. the field of the mirror
     generator `m(g)` at the Tx position (`P1(:, g) = P0(:, m(g))`; `F_m(g)` equals `F_g` scaled by
     `lambda'_g / lambda'_m(g)`) [W-2]. For the self-mirrored generators this is `g`'s own field at the Tx position.
   Generators that share `F` (same `lambda'`) may be evaluated in one `sos_val` call with several phase columns
   (same values).
5. Combination, per link `l`, with `D = norm(R(l) - T(l))` (projected positions in 2D mode) and
   `A_g(D) = qd_sos_acf_interp(acf_g, dist_g, D)` (linear, constant beyond the last sample point) [R1-23]:
   - single mobility: `x_g = a_g`;
   - dual mobility, `m(g) = g`: `x_g = (a_g + b_g) / sqrt(2 + 2·A_g(D))` (both devices on one map; unit variance);
   - dual mobility, pairs (3, 4) and (5, 6): `x_g = (a_g + b_g) / sqrt(2)` (two independent fields), then the
     coupling of the pair `(g, m)` [W-3]:
   ```
   c   = 0.5 · (A_g(D) + A_m(D))                      symmetric correlation of the two devices' fields
   c   = min(c, 1 - 1e-5)                             as gen_lsf_parameters.m
   s   = sqrt(1 - c²);  t = sqrt(2 + 2·s);  p = (1 + s) / t;  q = c / t
   (x_g, x_m) <- (p·x_g + q·x_m, q·x_g + p·x_m)      principal square root of [[1, c], [c, 1]]
   ```
6. `rnd` `[n_link, 11]`: column g = `x_g` for g = 0..8; `0.5 · erfc(-x_g / sqrt(2))` for g = 9, 10.

This is equivalent to the device modes of `sos_val` (two devices on one map for the self-mirrored generators, two
maps with `freq_scale_b = lambda'_g / lambda'_m(g)` for the pairs) but evaluates each field with its own frequency
table, so that swapping `tx_pos` and `rx_pos` repeats exactly the same `sos_val` calls in swapped roles: the swap
reciprocity holds bit for bit, also for unequal decorrelation distances and with bias-removed phases.

**Properties to document** (and test): columns 0 to 8 Normal(0,1), 9 and 10 Uniform(0,1); the value of a link does
not depend on the other links of the call; 2D mode: the values do not depend on z; dual mobility: swapping
`tx_pos` and `rx_pos` leaves columns 0, 1, 2, 7, 8, 9, 10 unchanged and swaps 3 with 4 and 5 with 6, also for
unequal decorrelation distances and with bias removal [W-2, W-5]; floors: fields of different floor IDs are
uncorrelated at equal horizontal positions (3.2.4); precision: the sinusoids are evaluated with the AVX2 kernel
where available (single-precision sin/cos, deviation below `4e-7 · sqrt(2 · n_coeff)`) [W-44]; `preset` and
`horizontal` must be those of `lsf_init_phases` (a preset with the same `n_coeff`, e.g. `Exp256` for `Comb256`,
cannot be detected).

#### 3.2.4 Floor phase offsets [W-1]

```
N = n_coeff;  q = floor ID (arma::uword);  k = q mod N;  b = q div N
Delta_q(n) = 2·pi · k · n / N + psi_b(n),        n = 0 .. N-1
psi_0(n)   = 0
psi_b(n)   = 2·pi · u_n                         b >= 1; rng = qd_sos_rng(b), u_n = qd_sos_uniform(rng), n = 0 .. N-1 in order
```

`qd_sos_rng(b)` seeds `std::mt19937_64` with `b`, and `qd_sos_uniform` maps one raw 64-bit output to
`(x >> 11) · 2^-53` [R1-23]; both are portable across standard libraries (unlike
`std::uniform_real_distribution`), so a test can retype the rule with `std::mt19937_64 gen(b)`.

Properties: for two floors in the same block of N IDs the cross-correlation of their fields at zero horizontal lag
(the same x, y) is `(1/N) · sum_n cos(Delta_q(n) - Delta_q'(n)) = 0` exactly (k differs, the sum over a full period
of a discrete cosine vanishes); for floors in different blocks it is a random number with standard deviation
`1/sqrt(2N)` (0.044 for 256 sinusoids). At non-zero horizontal lag the cross-covariance of two floors is not zero
(up to 0.11 to 0.16 for the 256-sinusoid presets, 0.07 to 0.10 for 1024, at lags of 0 to 60 m in 2D mode), the
same level as for pseudo-random offsets [R1-16]; the documentation block states "uncorrelated at equal horizontal
positions", not "independent". The same offsets serve both slices and all generators, so mirror relations and
reciprocity are kept per floor. The offsets are file-local in `src/lsf_draw.cpp` (a lambda or a function in an
anonymous namespace).

### 3.3 `path_loss`

Source: `get_pl.m` (complete), TR Table 7.4.1-1 and note 1, clause 7.4.3.1; decisions D2, D4, D16; plan P1 to P6.

#### 3.3.1 Validation (after 3.0.2)

1. `d3d_in` (if given): `n_link` elements; finite and >= 0: `Input 'd3d_in' must be finite and >= 0.`
2. `rnd_he` (if given): `n_link` elements; values in [0, 1]: `Input 'rnd_he' must be in the range from 0 to 1.`
3. `pl_model`: empty, or one of the 10 names in 3.3.4: `Path-loss model '<name>' is not supported.`
4. Required parameters of the model (3.3.4) present: `Path-loss model '<model>' requires the parameter '<key>'.`
   Every `plpar` entry that the model reads (required or optional) must have exactly one value:
   `Path-loss parameter '<key>' must have one value.` [W-9] (no conf file in `data/config` or in QuaDRiGa-NG has a
   `PL_` key with several values). Entries not read by the model are ignored.
5. Breakpoint factors: `E` (dual_slope; nlos with LOS term), `E1`, `E2` (tripple_slope) must be > 0:
   `Path-loss parameter '<key>' must be > 0.`; `winner_los`: `A1 != A2`, else
   `Path-loss parameters 'A1' and 'A2' must differ for model 'winner_los'.` [W-9]
6. scenpar completed and validated (3.0.3; it holds `LSP_fc_min` and the SF keys of the default `sf_sigma`).

#### 3.3.2 Distances

Per link (frequency-independent):

```
d3 = d3D,  d2 = d2D                                  geometric distances (TR: d3D-out + d3D-in, eq. 7.4-1)
if d3d_in given:                                      legacy outdoor distance, get_pl.m lines 177 to 186 [W-46]
    p  = d3 > 0 ? d3d_in(l) / d3 : 0                  [W-35]
    d3 = d3 - d3d_in(l)
    d2 = d2 · (1 - p)
d3 = max(d3, 0.1);  d2 = max(d2, 0.1)                 get_pl.m lines 189, 190
```

Without `d3d_in` (TR, default) the geometric distance between the Tx and the indoor Rx is used; it is already
`d3D-out + d3D-in`. Passing `d3d_in` selects the legacy QuaDRiGa mode, in which the path-loss formulas see only the
outdoor part [W-46]. The O2I loss itself is never added by `path_loss`; the caller adds the output of `o2i_loss`
[I-7]. [pending user decision R1-10: with the reduced O2I scope the legacy input becomes the indoor 2D distance
`d2d_in` per link, and the geometry of 3.6.2 step 5 (d3D-in from d2D-in, minimum outdoor distance 10 m) moves here
in front of the three lines above.]

`f` is not clamped by `LSP_fc_min` in the path loss.

#### 3.3.3 Default shadow-fading standard deviation [W-8]

`sf_sigma(l, k) = sigma_2(l, k)` of 3.0.4 (SF index, raw geometry, not the O2I-scaled distances, `vs_ratio`
irrelevant). Models with `sig`-type keys replace it as stated below. For all v19 files without `sig` keys this is
`SF_sigma` (they have `SF_delta = SF_omega = SF_kappa = SF_tau = SF_beta = 0`).

#### 3.3.4 Models

Notation: `[X]` is the first value of `plpar["X"]`; `[X|v]` is `[X]` if present, else `v`. "required" keys throw if
missing. `PL1(x) = A1·lg(x) + B + C·lg(f) + D·x`. Heights before clamping: `hBS`, `hUT` (raw z).

| Model | Required keys | Defaults |
|---|---|---|
| (empty) | none | `pl = 0`, `sf` default |
| `constant` | A | |
| `logdist` | A, B, C | SF optional |
| `logdist_simple` | A, B | |
| `winner_los` | A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, F1, F2, sig1, sig2 | |
| `winner_nlos` | A1, A2, A3, Ah1, Ah2, Ah3, B1, B2, B3, C1, C2, C3, D1, D2, D3, E1, E2, E3, F1, F2, F3 | |
| `winner_pathloss` | A, B, C, X | SF optional |
| `dual_slope` | A1 or A, B, E | A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, BP2D = 0, sig1/sig2 optional |
| `tripple_slope` | A1 or A, B, E1 or E | A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0, sig1..sig3 optional |
| `nlos` | An, Bn; plus the `dual_slope` keys if LOSmax != 0 | Cn = Dn = D1n = D2n = D3n = En = E1n = E2n = E3n = Fn = G1n = 0, G2n = 1, LOSmax = 1, Ax = Bx = Cx = 0 (term inactive unless one of them is present), sig optional |
| `satellite` | none | A = 20, B = 32.45, C = 20, D = 0, usePLa = 1 |

Formulas per link `l` and frequency `k` (`get_pl.m` line numbers in brackets):

- **constant** [210]: `pl = [A]`.
- **logdist** [201]: `pl = [A]·lg(d3) + [B] + [C]·lg(f)`; `sf = [SF]` if present.
- **logdist_simple** [207]: `pl = [A]·lg(d3) + [B]`.
- **winner_los** [213 to 245]: `hb = max(hBS, 1.5)`, `hm = max(hUT, 1.5)` of the link itself [W-32];
  `G = B1 + C1·lg f + D1·lg hb + E1·lg hm`, `H = B2 + C2·lg f + D2·lg hb + E2·lg hm`,
  `bp = 10^((H - G)/(A1 - A2))`;
  `d2 <= bp`: `pl = A1·lg d2 + B1 + C1·lg f + D1·lg hb + E1·lg hm + F1·hm`, `sf = sig1`;
  else `pl = A2·lg d2 + B2 + C2·lg f + D2·lg hb + E2·lg hm + F2·hm`, `sf = sig2` [W-42]. (`get_pl.m` takes the
  breakpoint from the mean heights of the call, i.e. of one track, and uses `C1` in the second branch, line 241;
  both differ from the library only where the heights vary within one MATLAB call, or beyond the breakpoint when
  `C1 != C2`. With `C2` the loss is continuous at `bp`: `A2·lg bp + H + F2·hm = A1·lg bp + G + F1·hm` holds when
  `F1 = F2`, as in all WINNER files.)
- **winner_nlos** [247 to 278]: `hb = max(hBS, 1.5)`, `hm = max(hUT, 1.5)`;
  `L_j = (A_j + Ah_j·lg hb)·lg d2 + B_j + C_j·lg f + D_j·lg hb + E_j·lg hm + F_j·hm` (j = 1, 2, 3);
  `pl = L_1` for `f < 1.5`, `L_2` for `1.5 <= f < 2`, `L_3` for `f >= 2`; `sf` default.
- **winner_pathloss** [280 to 289]: `pl = [A]·lg d3 + [B] + [C]·lg(f/5) + [X]`; `sf = [SF]` if present.
- **dual_slope** [291 to 323]: `hE` = `[hE|0]`, or per link from 3.3.6 if `[hE_rand|0] != 0` and `rnd_he` is given;
  `hb = hBS; if hb < 1.1·hE: hb = 1.1·hE; if hb < 0.1: hb = 0.1` (same for `hm` from `hUT`);
  `dBP = E·f·(hb - hE)·(hm - hE)`; `dBP3 = sqrt(dBP² + (hb - hm)²)`;
  `d2 <= dBP`: `pl = PL1(d3)`, `sf = [sig1]` if present; `d2 > dBP`: `pl = PL1(dBP3) + A2·lg(d3/dBP3)` (3D
  breakpoint distance; TR UMa, UMi, InH, InF), or with `[BP2D|0] != 0` `pl = PL1(dBP) + A2·lg(d3/dBP)` (2D
  breakpoint distance; TR RMa and SMa LOS, Table 7.4.1-1) [W-41]; `sf = [sig2]` if present. (The two forms differ
  by `(A1 - A2)·lg(dBP3/dBP) + D·(dBP3 - dBP)`, up to 0.034 dB for RMa/SMa in the TR's height and frequency
  ranges.)
- **tripple_slope** [325 to 378]: `hb = max(hBS, 1.1·hE1, 1.1·hE2, 0.1)` (same for `hm`);
  `dBP1 = E1·f·(hb - hE1)·(hm - hE1)`, `dBP2 = E2·f·(hb - hE2)·(hm - hE2)`, `dBP1 = min(dBP1, dBP2)`;
  `dBPj3 = sqrt(dBPj² + (hb - hm)²)`; `LBP1 = PL1(dBP13)`, `LBP2 = LBP1 + A2·lg(dBP23/dBP13)`;
  `d2 <= dBP1`: `PL1(d3)`, `sf = sig1`; `dBP1 < d2 <= dBP2`: `LBP1 + A2·lg(d3/dBP13)`, `sf = sig2`;
  `d2 > dBP2`: `LBP2 + A3·lg(d3/dBP23)`, `sf = sig3` (each `sf` only if the key is present).
- **nlos** [380 to 434] plus P2, P3: `hb = max(hBS, 0.1)`, `hm = max(hUT, 0.1)`;
  ```
  PLn = An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² + D3n·hb
      + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²
  pl  = PLn
  if [LOSmax|1] != 0:      pl = max(pl, PL_dual_slope)     (dual_slope above with the same plpar incl. hE_rand and BP2D, its sf unused)
  if Ax, Bx or Cx present: pl = max(pl, [Ax|0]·lg d3 + [Bx|0] + [Cx|0]·lg f)
  sf = [sig] if present
  ```
- **satellite** [436 to 476]: `sa = min(hBS / d3, 1)` (absolute Tx height, as MATLAB); if `sa <= 0`: `pl = +Inf`
  [W-10]; else `alpha = asin(sa)` [rad], `PLa = [usePLa] != 0 ? Z(f) / sa : 0` with `Z(f)` the zenith attenuation
  of Appendix 11.1, linearly interpolated over f in GHz with constant values beyond 1 and 100 GHz
  (`qd_sos_acf_interp(Z, f_grid, 93, f)`: linear, constant beyond both ends, strictly increasing grid) [R1-23];
  `pl = A·lg d3 + B + C·lg f + D·lg(alpha) + PLa`.

`sf_sigma` is the default (3.3.3) wherever a model does not set it.

#### 3.3.5 Edge rules

- `n_link = 0`: empty outputs (`winner_los` computes no mean).
- `fc` may be any positive frequency; the models are evaluated as written, also outside their validity range.
- Missing `PL_hE_rand`, `PL_LOSmax`, `PL_Ax/Bx/Cx`, `PL_BP2D` keep today's behaviour [W-40]; `PL_hE_rand = 1`
  without `rnd_he` uses the fixed `PL_hE` (plan 6.5).

#### 3.3.6 UMa random environment height (TR Table 7.4.1-1 note 1) [D15, W-11, W-36]

Per link, frequency-independent, in `dual_slope` (and the LOS term of `nlos`) when `[hE_rand|0] != 0` and `rnd_he`
is given; `u = rnd_he(l)`, `hUT` raw, `d` = the `d2` of 3.3.2:

```
g  = (d <= 18) ? 0 : 1.25 · (d/100)³ · exp(-d/150)
C' = (hUT < 13) ? 0 : (hUT <= 23 ? ((hUT - 13)/10)^1.5 : 1)
C  = C' · g;   p1 = 1 / (1 + C)
n  = (hUT - 1.5 >= 12) ? floor((hUT - 13.5)/3 + 1e-9) + 1 : 0          size of the set {12, 15, ..., hUT - 1.5}
if C == 0 or n == 0 or u < p1:  hE = 1
else:  j = min(floor((u - p1) / (1 - p1) · n), n - 1);  hE = 12 + 3·j
```

The fixed `[hE]` is not used for such a link. The value 1 m is the TR value (the v19 UMa files also have `PL_hE = 1`).

#### 3.3.7 Documentation block (must state)

The 10 models with their formulas or a pointer to the parameter names; the distance clamps; the O2I distance rule
(geometric distance without `d3d_in`, outdoor distance with it) and that the O2I loss is not added; the default
`sf_sigma`; the random hE rule; the new keys `PL_hE_rand`, `PL_LOSmax`, `PL_Ax`, `PL_Bx`, `PL_Cx`, `PL_BP2D`; that
`f` is not clamped; required keys, the one-value rule and the error for a missing key; `[n_link, n_freq]` sizes.
`winner_los` is documented with `C2` beyond the breakpoint and per-link breakpoint heights (no reference to the
MATLAB behaviour in the block).

### 3.4 `lsp_generate`

Source: `gen_lsf_parameters.m` (lines 228 to 268 and 309 to 361), `qd_builder.m` `lsp_vals`; TR 7.5
step 4, Tables 7.5-6 to 7.5-12, 7.6.5, 7.6.9; decisions D5, D16, D18, D22, D26; plan L1 to L9, T1, T2, R1.

#### 3.4.1 Validation (after 3.0.2)

1. `rnd`: `[n_link, 11]`: `Input 'rnd' must have size [n_link, 11].`; finite: `Input 'rnd' must be finite.`
2. `sf_sigma` (if given): `[n_link, n_freq]`: `Input 'sf_sigma' must have size [n_link, n_freq].`; finite and
   >= 0: `Input 'sf_sigma' must be finite and >= 0.` (a negative value would flip the sign of the SF) [R1-25].
3. `vs_ratio` finite and >= 0: `Input 'vs_ratio' must be finite and >= 0.`
4. scenpar completed and validated with `lsp_xcorr` and the return value (3.0.3); this covers `LSP_fc_min`,
   `DS_VS`, `DS_VS_a`, `DS_VS_b`, `absTOA_Lmax` and the cross-key rule of 4.2 (`DS_VS > 0` with
   `DS_VS_a = DS_VS_b = 0`). Function-specific: `vs_ratio > 0` with `DS_VS_a = DS_VS_b = 0`:
   `Input 'vs_ratio' requires 'DS_VS_a' or 'DS_VS_b' to be > 0.` [W-14]
5. Not positive definite (return value of `scenpar_validate` false):
   `LSP cross-correlation matrix in 'scenpar' is not positive definite.` (both roots; MATLAB refuses as well,
   `gen_lsf_parameters.m` lines 120 to 123).
6. Only if `gr_eps_re` or `gr_eps_im` is requested: column 9 of `rnd` in [0, 1]:
   `Column 9 of 'rnd' must be in the range from 0 to 1.` [W-37]

#### 3.4.2 Algorithm

1. Root `Q` (8x8, our order DS, KF, SF, ASD, ASA, ESD, ESA, XPR) of `R = lsp_xcorr`:
   - `cholesky = true` (TR step 4, order `[SF, K, DS, ASD, ASA, ZSD, ZSA]`, XPR last): permutation
     `pi = [2, 1, 0, 3, 4, 5, 6, 7]` (TR position a holds our LSP `pi[a]`); `Rp(a, b) = R(pi[a], pi[b])`;
     `qd_gbsm_chol8(Rp, L)` (lower, `Rp = L·L^T`); `Q(pi[a], pi[b]) = L(a, b)`. Then `x = Q·s` equals the TR
     rule `x_TR = L·s_TR` with `s_TR[a] = s[pi[a]]`. In particular `x_SF = s_SF` (the SF field is passed through).
   - `cholesky = false` (legacy, MATLAB `sqrtm`): `Q` = symmetric square root of `R` (3.4.3, file-local)
     [pending user decision R1-09].
2. Per link: `s = rnd(l, 0..7)^T`, `x = Q·s` (same `x` for all frequencies [L9, TR 7.6.5]).
3. Per link and frequency: `mu_i`, `sigma_i` from 3.0.4 (with `vs_ratio`); if `sf_sigma` is given,
   `sigma_2 = sf_sigma(l, k)` [D16]. `v_i = mu_i + sigma_i · x_i`.
   `lsp(l, k, i) = 10^(v_i)` for i in {0, 3, 4, 5, 6}; `10^(0.1·v_i)` for i in {1, 2, 7}.
4. `clip_spreads = true` [L2]: `lsp(:, :, 3) = min(., 104)`, `lsp(:, :, 4) = min(., 104)`,
   `lsp(:, :, 5) = min(., 52)`, `lsp(:, :, 6) = min(., 52)`. `lsp_mu`, `lsp_sigma` are not affected.
5. `lsp_mu(l, k, i) = mu_i`, `lsp_sigma(l, k, i) = sigma_i` (with the `sf_sigma` replacement).
6. `abs_toa(l)` [D22, T1, T2]: if `absTOA_mu > -30`:
   `min(10^(absTOA_mu + absTOA_sigma · rnd(l, 8)), absTOA_Lmax > 0 ? 2·absTOA_Lmax/c : +Inf)`; else 0.
7. Ground permittivity [R1], `gen_lsf_parameters.m` lines 319 to 344, `f` unclamped, `U = rnd(l, 9)`:
   ```
   dry = 3            + j·0.003·f^1.34
   med = 30.4·f^-0.47 + j·0.18·f^1.05
   wet = 31.3·f^-0.48 + j·0.63·f^0.77
   U <= 0.5:  w = 2·U;          eps = (1 - w)·dry + w·med
   U >  0.5:  w = 2·(U - 0.5);  eps = (1 - w)·med + w·wet
   gr_eps_re = Re(eps), gr_eps_im = Im(eps)     (imaginary part >= 0, QuaDRiGa convention [W-27])
   ```
   TR eq. (7.6-40) writes the complex permittivity as `eps_r - j·sigma/(2·pi·f·eps0)`, i.e. with the opposite sign
   of the imaginary part. A caller (the later SSF port) that uses the TR's reflection formulas takes the complex
   conjugate, `gr_eps_re - j·gr_eps_im` [R1-29].
8. EoD offset [L5], plan 5.1, with `f'`, `d`, `h` of 3.0.4 and raw `hUT`:
   ```
   e = EoD_offset_e + EoD_offset_e_gamma · lg f'
   if EoD_offset_b > 0:
       e += 10^( (EoD_offset_a + EoD_offset_a_gamma·lg f') · lg(max(EoD_offset_b, d))
                 + EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT · hUT )
   if EoD_offset_hcl > 0 and hUT < EoD_offset_hcl:
       e += (atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d)) · 180/pi
   eod_offset_deg(l, k) = e                       degrees, positive = up (= -mu_offset,ZOD of the TR)
   ```

#### 3.4.3 Square root by cyclic Jacobi (legacy root)

[pending user decision R1-09] This subsection stays as in r0 until Stephan decides whether the legacy root is
kept. The local edits for the two alternatives:
- Drop the option: delete this subsection, the `cholesky = false` bullet of 3.4.2 step 1, the argument `cholesky`
  in 2.1 and in the MEX/Python Usage lines (6.1, 6.2; MEX `nrhs 5..8`), the test row "the legacy root is the
  symmetric square root" and the "both roots" part of the sample-covariance row (5.4), the MEX/Python test point
  "`cholesky = false` changes the KF/SF mixing" (6.1), the row `lsp_generate cholesky` of the switch table in
  section 10, and in 10 (15d) add: NG computes `x = sqrtm(R)·s` itself and calls `lsp_generate` with a scenpar
  whose 28 correlation keys are 0 (identity root) and `rnd(:, 0..7) = x`.
- Keep the option but use Newton-Schulz instead of Jacobi: replace the Jacobi paragraph below by the coupled
  Newton-Schulz iteration on `A / tr(A)` (`Y0 = A/tr(A)`, `Z0 = I`; `T = (3I - Z·Y)/2`, `Y <- Y·T`, `Z <- T·Z`,
  until `||Y·Y - A/tr(A)||_F <= 1e-15·||A/tr(A)||_F`, at most 100 iterations; `S = sqrt(tr(A))·Y`, then
  `S = (S + S^T)/2`); the tests are unchanged.

The root is file-local in `src/lsp_generate.cpp` (single user) and tested through `lsp_generate` (5.4).

For the symmetric 8x8 `A`: `V = I`, `B = A`. Sweeps over all pairs `p < q` in row order: skip if
`abs(B(p,q)) <= 1e-300`; else `theta = (B(q,q) - B(p,p)) / (2·B(p,q))`,
`t = sign(theta) / (abs(theta) + sqrt(theta² + 1))` (`sign(0) = 1`), `cs = 1/sqrt(t² + 1)`, `sn = t·cs`; apply the
rotation `J` (identity with `J(p,p) = J(q,q) = cs`, `J(p,q) = sn`, `J(q,p) = -sn`) as `B = J^T·B·J`, `V = V·J`.
Stop when the off-diagonal Frobenius norm of `B` is `<= 1e-15 ·` the Frobenius norm of `A`, at the latest after 50
sweeps. `S = V·diag(sqrt(max(B(i,i), 0)))·V^T`, then `S = (S + S^T)/2`. For a symmetric positive definite `A` this
is the principal root that MATLAB's `sqrtm` returns (D23 parity). The Cholesky is `qd_gbsm_chol8` (3.0.5).

#### 3.4.4 Documentation block (must state)

Units of every slice [D5]; the mean/std formula incl. `LSP_fc_min`, the ESD terms and the floor, the V/S term and
`vs_ratio`; both roots and the TR order; clipping; `sf_sigma`; same random values for all frequencies; the abs TOA
rule and bound; the ground permittivity model and its sign convention, with the note that TR eq. (7.6-40) uses the
opposite sign of the imaginary part (take the conjugate for the TR reflection formulas); the EoD offset formula and
sign; that `sf_sigma` must be >= 0; that
`rnd = zeros(n_link, 11)` gives the median LSPs and the `lsp_mu` / `lsp_sigma` outputs without randomness
(replacement of `lsp_vals`, section 10).

### 3.5 `los_probability`

Source: TR Table 7.4.2-1, clause 7.6.3.3 eq. (7.6-18), 7.8.4 Table 7.8-7; `set_scenario.m` (lines
338 to 715); decisions D7, D8, D24.

#### 3.5.1 Validation (in this order)

1. `model` one of the 18 names of the header: `Model '<name>' is not supported.`
2. `d2d_out` finite and >= 0: `Input 'd2d_out' must be finite and >= 0.`; `n_link = d2d_out.n_elem` (0 allowed).
3. `h_bs`, `h_ut`: 1 or `n_link` elements (`Number of elements in 'h_bs' must be 1 or match 'd2d_out'.`), finite
   (`Input 'h_bs' must be finite.`). A single element applies to all links.
4. `d_clutter` NaN or finite > 0 (`Input 'd_clutter' must be NaN or > 0.`); `r_clutter` NaN or in [0, 1)
   (`Input 'r_clutter' must be NaN or in the range from 0 to 1, 1 excluded.`); `h_clutter` NaN or finite >= 0
   (`Input 'h_clutter' must be NaN or >= 0.`). Checked for every model, used only by InF and SMa.
5. If `los_soft != nullptr`: `rnd_g` and `fc` are required (`Output 'los_soft' requires 'rnd_g' and 'fc'.`);
   `rnd_g` `n_link` elements, finite; `fc` non-empty, finite > 0. If `los_soft == nullptr`, `rnd_g` and `fc` are
   ignored (not validated).

#### 3.5.2 Models (`d = d2d_out(l)`, `hB = h_bs`, `hU = h_ut`; all results limited to [0, 1] [W-15])

| Model | p_LOS |
|---|---|
| `38.901_RMa` | `d <= 10`: 1; else `exp(-(d - 10)/1000)` |
| `38.901_UMi` | `d <= 18`: 1; else `18/d + exp(-d/36)·(1 - 18/d)` |
| `38.901_UMa` | `d <= 18`: 1; else `(18/d + exp(-d/63)·(1 - 18/d)) · (1 + C'(hU)·5/4·(d/100)³·exp(-d/150))`, `C' = 0` for `hU <= 13`, `((hU - 13)/10)^1.5` for `13 < hU <= 23`, 1 for `hU > 23` [W-11] |
| `38.901_InH_Mixed` | `d <= 1.2`: 1; `1.2 < d < 6.5`: `exp(-(d - 1.2)/4.7)`; `d >= 6.5`: `exp(-(d - 6.5)/32.6)·0.32` |
| `38.901_InH_Open` | `d <= 5`: 1; `5 < d <= 49`: `exp(-(d - 5)/70.8)`; `d > 49`: `exp(-(d - 49)/211.7)·0.54` |
| `38.901_InF_SL`, `_DL` | `exp(-d·kinv)`, `kinv = -ln(1 - r)/d_clutter` |
| `38.901_InF_SH`, `_DH` | see below |
| `38.901_InF_HH` | 1 |
| `38.901_SMa` | `d <= 10`: 1; else see below |
| `38.811_DenseUrban`, `_Urban`, `_Suburban`, `_Rural` | `0.01 ·` linear interpolation of the table of Appendix 11.2 at the elevation `atan2(hB, d)` in degrees (`qd_sos_acf_interp` over the elevation grid) [W-18] |
| `QuaDRiGa_Industrial` | `min(1, exp(-(d + 20)/80) · exp(min(hB, 8)/5 - 1))` |
| `LOS` / `NLOS` | 1 / 0 [W-47] |

The TR 36.873 models of QuaDRiGa's 3GPP_3D scenarios are not separate names (R1-12): `36.873 UMi` equals
`38.901_UMi` (for `d > 18`: `(18/d)·(1 - e) + e = 18/d + e·(1 - 18/d)`, `e = exp(-d/36)`; 1 below), and
`36.873 UMa` with the TR 36.873 term `1.25e-6·d³·exp(-d/150) = 5/4·(d/100)³·exp(-d/150)` equals `38.901_UMa`
(36.873 applies the height term to indoor UTs only; its layouts have outdoor UTs at 1.5 m, where `C' = 0` as
well). NG maps these groups to the
38.901 names (section 10).

InF defaults (TR 7.8.4 calibration, plan S5), used where the argument is NaN:

| Model | r | d_clutter [m] | h_c [m] |
|---|---|---|---|
| SL | 0.2 | 10 | (unused) |
| DL | 0.6 | 2 | (unused) |
| SH | 0.2 | 10 | 2 |
| DH | 0.6 | 2 | 6 |

InF-SH, InF-DH [W-16]: `kinv0 = -ln(1 - r)/d_clutter`;
- `hU >= h_c`: p = 1 (UT above the clutter);
- else if `hB > hU`: `p = exp(-d · kinv0 · (h_c - hU)/(hB - hU))` (TR: `k = -d_clutter/ln(1 - r) · (hB - hU)/(h_c - hU)`);
- else (`hB <= hU < h_c`, BS inside the clutter below the UT): `p = exp(-d · kinv0)` (the SL/DL rule).
`r = 0` gives `kinv0 = 0` and p = 1.

SMa (TR v19), `d > 10`, defaults `d_clutter = 30`, `r_com = 0.02`, `r_res = 0.18`, `r_veg = r_clutter` (default 0),
heights `h_com = 20`, `h_res = 8`, `h_veg = 15` (fixed); `h_clutter` is ignored [W-16]:
```
if hB <= hU:  p = 1
else:  kinv_x = -ln(1 - r_x)/d_clutter · max(h_x - hU, 0)/(hB - hU)       x = com, res, veg
       p = exp(-d · (kinv_com + kinv_res + kinv_veg))
```
(The TR has `max(., 0)` only for the residential term; for `hU < 15 m`, the TR range, the results are equal.)

#### 3.5.3 Soft LOS state (TR eq. 7.6-18) [W-20, W-21]

```
F(p)            = probit(p) = sqrt(2) · erfinv(2p - 1)          p = p_LOS(l) after the [0, 1] limit
lambda_k        = c / fc(k)                                      wavelength [m]
los_soft(l, k)  = 0.5 + atan( sqrt(20 / lambda_k) · (rnd_g(l) + F(p)) ) / pi
```
`p = 1` gives `F = +Inf` and `los_soft = 1`; `p = 0` gives `F = -Inf` and `los_soft = 0` (also for any finite
`rnd_g`). `los_soft > 0.5` exactly when `rnd_g > -F(p)`, which has probability `p` for `rnd_g ~ N(0,1)`.

#### 3.5.4 Inverse normal CDF (file-local in `src/los_probability.cpp`) [W-21]

`p <= 0`: `-Inf`; `p >= 1`: `+Inf`. Else Acklam's approximation with coefficients

```
a = {-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00}
b = {-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01}
c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
     -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00}
d = { 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,  3.754408661907416e+00}
p_low = 0.02425
p < p_low:          q = sqrt(-2 ln p);      x = (((((c0 q + c1) q + c2) q + c3) q + c4) q + c5) / ((((d0 q + d1) q + d2) q + d3) q + 1)
p > 1 - p_low:      q = sqrt(-2 ln(1 - p)); x = -((((((c0 q + c1) q + c2) q + c3) q + c4) q + c5) / ((((d0 q + d1) q + d2) q + d3) q + 1))
else:               q = p - 0.5; r = q²;     x = (((((a0 r + a1) r + a2) r + a3) r + a4) r + a5) q / (((((b0 r + b1) r + b2) r + b3) r + b4) r + 1)
```
followed by one Halley step: `e = 0.5·erfc(-x/sqrt(2)) - p`, `u = e·sqrt(2·pi)·exp(x²/2)`,
`x = x - u/(1 + x·u/2)` (`std::erfc`). Accuracy target (what `los_soft` needs, with margin): `abs(x - F(p)) <=
1e-10 · max(1, abs(F(p)))` for `p` in `[1e-12, 1 - 1e-12]`, `F` the exact inverse of `Phi(x) = 0.5·erfc(-x/sqrt(2))`
(Acklam alone reaches about 1e-9 relative, one Halley step about machine precision). The probit has one user and
is a lambda or an anonymous-namespace function of `los_probability.cpp`; it is tested through `los_probability`
(5.5).

#### 3.5.5 Documentation block (must state)

Every model with its formula and source (TR 38.901 Table 7.4.2-1, TR 38.811 via QuaDRiGa, QuaDRiGa); that the
TR 36.873 UMi/UMa models equal `38.901_UMi` / `38.901_UMa`; the defaults of `d_clutter`, `r_clutter`, `h_clutter`;
the edge rules; the [0, 1] limit; the soft LOS formula; that the LOS state draw itself and its correlation distance
(`LOS_state_lambda`) are the caller's; that for O2I UTs `d2d_out = d2D - d2D-in` (from `o2i_loss`, run first),
clamped at 0 by the caller (negative distances are rejected) [R1-20]; that the `LOS_state_lambda` of the InF files
is `d_clutter / 2` of the TR defaults and does not follow a `d_clutter` argument (the caller passes its own
correlation distance when it changes `d_clutter`).

### 3.6 `o2i_loss`

Source: TR 7.4.3.1 (Tables 7.4.3-1 to 7.4.3-3), 7.4.3.2, 7.6.3.3; `gen_o2i_loss.m` (lines 110 to
308); decisions D25; plan O1 to O8.

#### 3.6.0 Scope [pending user decision R1-10]

The text below is the r0 scope (D25: all of d3D-in geometry, mmMAGIC, car loss, < 6 GHz model). Stephan decides
whether it is reduced to the TR models. The local edits for the reduced scope:
- Models: `low`, `high`, `low_A`, `legacy_6GHz`, `car`, `car_metal` only (delete the rows `high_v14`,
  `mmMAGIC_low`, `mmMAGIC_high`, the term `L_IRR_v14` and the parameters `alpha`, `d_room`).
- `rnd` always `[n_ut, 4]` (validation 2: `Input 'rnd' must have 4 columns.`; delete the 6-column rules).
- Delete the inputs/outputs `tx_pos`, `rx_pos`, `d3d_in`, `loss_el` and the flag `d2d_in_min_of_two`, validation
  items 6 and 7, algorithm step 5, the legacy rule of step 2 and Appendix 11.3.
- `path_loss`: the optional legacy input becomes `const arma::vec *d2d_in` (indoor 2D distance per link); the d3D-in
  geometry of step 5 (from `din = min(d2d_in, d2)`, minimum outdoor distance 10 m) moves into 3.3.2 in front of
  the legacy distance lines.
- Tests 5.6 lose the rows of the deleted models, the 6-column cases, the geometry row and the legacy d2D-in part;
  wrappers 6.1/6.2 lose the deleted arguments and outputs (MEX `nrhs 4..5`, `nlhs <= 3`); section 10 loses the
  switches `d2d_in_min_of_two` and `high_v14` and states that NG keeps mmMAGIC in MATLAB if needed.

#### 3.6.1 Validation (in this order)

1. `fc` (3.0.2).
2. `rnd`: 4 or 6 columns: `Input 'rnd' must have 4 or 6 columns.`; 6 columns if `models` contains `mmMAGIC_low` or
   `mmMAGIC_high`: `Input 'rnd' must have 6 columns for the mmMAGIC models.`; columns 0, 1, 2, 4, 5 in [0, 1]:
   `Uniform columns of 'rnd' must be in the range from 0 to 1.`; column 3 finite: `Column 3 of 'rnd' must be
   finite.` `n_ut = rnd.n_rows` (0 allowed).
3. `models` non-empty (`Input 'models' cannot be empty.`); each name one of the 9 of the header
   (`Model '<name>' is not supported.`).
4. `weights`: as many elements as `models` (`Number of elements in 'weights' does not match number of models.`);
   finite and >= 0 (`Input 'weights' must be finite and >= 0.`); `abs(sum - 1) <= 1e-6`
   (`Input 'weights' must sum to 1.`).
5. `d2d_in_max` finite and >= 0: `Input 'd2d_in_max' must be finite and >= 0.`
6. `tx_pos`, `rx_pos`: either both given or both `nullptr` (`Inputs 'tx_pos' and 'rx_pos' must be given
   together.`); `tx_pos` 3 columns, at least 1 row, finite; `rx_pos` `[n_ut, 3]` (`Input 'rx_pos' must have size
   [n_ut, 3].`), finite.
7. `d3d_in` or `loss_el` requested without positions: `Output 'd3d_in' requires 'tx_pos' and 'rx_pos'.` /
   `Output 'loss_el' requires 'tx_pos' and 'rx_pos'.`

#### 3.6.2 Algorithm (per UT `u`, `f` in GHz per frequency)

1. Model choice [W-19]: `Ccum_j = (w_0 + ... + w_j) / sum(w)`; model `j*` = smallest `j` with `rnd(u, 0) < Ccum_j`;
   if there is none (u >= the last cumulative value), the last `j` with `w_j > 0`. (A weight of 0 is never chosen.)
2. Indoor distance (building models, i.e. all but `car`, `car_metal`):
   - `legacy_6GHz`: `d2din = rnd(u, 1) · d2d_in_max` (one uniform, Table 7.4.3-3).
   - otherwise, `d2d_in_min_of_two = true` (TR): `d2din = min(rnd(u, 1), rnd(u, 2)) · d2d_in_max`;
     `false` (legacy, O8): `d2din = Finv(rnd(u, 1)) · d2d_in_max` with `Finv` the linear interpolation of the
     legacy table (Appendix 11.3) from the CDF values to the distance fractions (`qd_sos_acf_interp` with the
     fractions as values and the CDF values as grid) [R1-23]. (R1-10 reduced: delete the legacy rule.)
   - car models: `d2din = 0`.
3. Loss without the elevation term, per frequency:
   ```
   L_glass = 2 + 0.2 f;  L_IRR = 25.4 + 0.11 f;  L_IRR_v14 = 23 + 0.3 f;  L_concrete = 5 + 4 f;  L_plywood = 1.03 + 0.17 f
   tw(p1, L1, p2, L2) = 5 - 10·lg(p1·10^(-L1/10) + p2·10^(-L2/10))
   ```
   | Model | PL_tw | PL_in | deviation |
   |---|---|---|---|
   | `low` | `tw(0.3, L_glass, 0.7, L_concrete)` | `0.5·d2din` | `4.4 · rnd(u, 3)` (N(0,1), not uniform as in MATLAB [I-6]) |
   | `high` | `tw(0.7, L_IRR, 0.3, L_concrete)` | `0.5·d2din` | `6.5 · rnd(u, 3)` |
   | `low_A` | `tw(0.3, L_glass, 0.7, L_plywood)` | `0.5·d2din` | `4.4 · rnd(u, 3)` |
   | `high_v14` | `tw(0.7, L_IRR_v14, 0.3, L_concrete)` | `0.5·d2din` | `6.5 · rnd(u, 3)` |
   | `legacy_6GHz` | 20 | `0.5·d2din` | 0 |
   | `car` | 0 | 0 | `9 + 5 · rnd(u, 3)` |
   | `car_metal` | 0 | 0 | `20 + 5 · rnd(u, 3)` |
   | `mmMAGIC_low` | `tw(0.3, L_glass, 0.7, L_concrete)` | `alpha·max(d2din - d_room, 0)` | `(4 + 0.1 f) · rnd(u, 3)` |
   | `mmMAGIC_high` | `tw(0.7, L_IRR_v14, 0.3, L_concrete)` | `alpha·max(d2din - d_room, 0)` | `(4 + 0.1 f) · rnd(u, 3)` |

   with `alpha = 0.5 + rnd(u, 4)`, `d_room = 2 + 2·rnd(u, 5)`. `loss(u, k) = PL_tw + PL_in + deviation`.
4. `d2d_in(u) = d2din`; `model_index(u) = j*` (0-based).
5. Geometry (only with `tx_pos`, `rx_pos`; `gen_o2i_loss.m` lines 185 to 223 and 262), per UT u and Tx t:
   ```
   d3 = norm(rx(u) - tx(t));  d2 = horizontal distance;  dz = rx(u).z - tx(t).z
   din = min(d2din, d2)
   d3in = d2 > 0 ? d3 · din / d2 : d3                        (MATLAB: NaN -> Inf -> d3)
   d3in = min(d3in, d3)
   if d3 - d3in < 10:  d3in = d3 - 10                        minimum outdoor distance 10 m
   d3in = max(d3in, 0)
   d3d_in(u, t) = d3in
   loss_el(u, t) = mmMAGIC model ? 20 · abs(theta) / 90 : 0,   theta = atan2(dz, d2) in degrees
   ```

#### 3.6.3 Documentation block (must state)

The 9 models with their table formulas (and that `high` uses the v19 IRR glass 25.4 + 0.11 f, `high_v14` the TR
v14 value 23 + 0.3 f of QuaDRiGa), the column meaning of `rnd`, the model choice rule, both d2D-in rules, the
geometry of `d3d_in`, that `loss` excludes the mmMAGIC elevation term (`loss_el`), that the caller draws `rnd` with
`sos_rand` / `sos_randn` at the TR correlation distances (building type 50 m, d2D-in 25 m, sigma_P 10 m, clause
7.6.3.3) and adds the result to the path loss of the indoor UT; RMa uses `low` only, InF `high` only, the backward
compatible model is `legacy_6GHz` with `O2I`-file `SF_sigma = 7` (the UMa/UMi O2I files already have 7 dB).

### 3.7 The moved functions

`conf_file_read`, `scenpar_validate`, `conf_file_list`, `conf_file_write`: no change of behaviour except the six new
keys (4.1), the V/S term in the DS overview line of `conf_file_write` (4.1) and the one new cross-key check of 4.2. Declarations move to `quadriga_gbsm.hpp` [D10]; the section of
their documentation blocks (C++, MEX, Python) becomes "Geometry-based stochastic model" [D11]; the key counts in the
documentation change from 165 to 171 numeric keys ("138 keys of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0"),
166 to 172 MEX fields / Python dict entries; Python moves them from `quadriga_lib.tools` to `quadriga_lib.gbsm`
[D29].


## 4. Conf-key changes

### 4.1 New numeric keys (R1)

Appended to the key table `qd_scenpar_keys[]` in `src/qd_scenpar.cpp` after `PolVar_sigma`, in this order, under the
comment line given below (it describes the content, not the project history [R1-26]). Defaults reproduce today's behaviour;
class `QD_NONNEG` (finite and >= 0); `row = col = -1`. Writer section (`sec`) and paragraph (`par`) as below
[W-34]; the writer sorts by paragraph, then table order, so the keys land after the existing keys of the same
paragraph.

```cpp
    // TR 38.901 v19.5.0: LOS/indoor state correlation distances, maximum indoor distance, InF delay-spread terms
    {"LOS_state_lambda", 0.0, QD_NONNEG, -1, -1, 4, 2, "decorrelation distance of the LOS/NLOS state [m]; 0 = uncorrelated"},
    {"IO_state_lambda", 0.0, QD_NONNEG, -1, -1, 4, 2, "decorrelation distance of the indoor/outdoor state [m]; 0 = uncorrelated"},
    {"O2I_d2D_in_max", 0.0, QD_NONNEG, -1, -1, 3, 4, "maximum indoor 2D distance of O2I UTs [m]; 0 = not stated"},
    {"DS_VS", 0.0, QD_NONNEG, -1, -1, 0, 1, "volume-to-surface ratio V/S of the hall [m]; 0 = no V/S term"},
    {"DS_VS_a", 0.0, QD_NONNEG, -1, -1, 0, 1, "factor a of the DS term log10(a V/S + b) [1/m]"},
    {"DS_VS_b", 0.0, QD_NONNEG, -1, -1, 0, 1, "offset b of the DS term log10(a V/S + b)"},
```

| Key | Meaning | Used by |
|---|---|---|
| `LOS_state_lambda` | Decorrelation distance of the LOS/NLOS state, Table 7.6.3.1-2 | caller (NG `set_scenario`), `sos_rand` |
| `IO_state_lambda` | Decorrelation distance of the indoor/outdoor state, Table 7.6.3.1-2 | caller |
| `O2I_d2D_in_max` | Maximum indoor 2D distance, clause 7.4.3.1 | caller passes it as `d2d_in_max` of `o2i_loss` |
| `DS_VS` | Reference V/S of the InF hall [D26] | `lsp_generate` (3.0.4) |
| `DS_VS_a`, `DS_VS_b` | `mu_lgDS = DS_mu + lg(a · V/S + b)` | `lsp_generate` |

After R1: `n_key = 171`; MEX struct and Python dict of `scenpar` have 172 entries (with `SubpathMethod`); the
documentation says "138 numeric keys of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0" and "then the 33 new
keys" (`src/qd_scenpar.cpp`, `api_mex/conf_file_*.cpp`, `api_mex/scenpar_validate.cpp`,
`api_python/qpy_conf_file_*.cpp`, `api_python/qpy_scenpar_validate.cpp`). The key-table comment at line 53 changes
accordingly.

Writer: no change for placement (the table entries carry section and paragraph; keys at their default are not
written, as for every key). One change in the overview formula of section 0 (`conf_file_write`, next to the ESD
special case): when `DS_VS > 0`, the DS mean is written as `log10( a * V/S + b )` followed by the usual mean
term with its sign, e.g. `% DS  = log10( 26 * V/S + 14 ) - 9.35 + Xds * ( 0.15 )` (a = `DS_VS_a`, b = `DS_VS_b`
written with `qd_conf_number`; `V/S` stays symbolic, its value is the `DS_VS` line); further terms (`DS_gamma`,
...) follow as today. With `DS_VS = 0` the line is unchanged. The `conf_file_write` documentation (C++, MEX,
Python) adds to the overview item: "the DS line has the term log10( a * V/S + b ) when DS_VS > 0". [R1-04]

### 4.2 Validation addition (R1)

In `qd_scenpar_check` (after the `NF_alpha`/`NF_beta` check), a cross-key check [W-14]:

```cpp
const arma::uword i_vs = qd_scenpar_find("DS_VS"), i_vsa = qd_scenpar_find("DS_VS_a"), i_vsb = qd_scenpar_find("DS_VS_b");
if (val[i_vs] > 0.0 && val[i_vsa] == 0.0 && val[i_vsb] == 0.0)
    throw std::invalid_argument(where(line == nullptr ? 0 : line[i_vs]) +
                                "Values of 'DS_VS_a' and 'DS_VS_b' must not both be 0 if 'DS_VS' is > 0.");
```

The `scenpar_validate` documentation lists the 6 keys under "Finite and >= 0" and the new rule under the cross-key
rules (C++, MEX, Python blocks).

### 4.3 New `PL_` keys (R3a)

No interface change (`plpar` is open, D7a). Behaviour in 3.3.4 and 3.3.6.

| Key | Model | Meaning | Default when absent |
|---|---|---|---|
| `PL_hE_rand` | dual_slope, nlos | non-zero: UMa random hE of note 1 from `rnd_he` | 0 (fixed `PL_hE`) |
| `PL_LOSmax` | nlos | 0: no maximum with the LOS term | 1 |
| `PL_Ax`, `PL_Bx`, `PL_Cx` | nlos | extra term `Ax·lg d3D + Bx + Cx·lg f` inside the maximum; active if one of them is present | inactive |
| `PL_BP2D` | dual_slope, nlos (LOS term) | non-zero: loss beyond the breakpoint `PL1(dBP) + A2·lg(d3D/dBP)` with the 2D breakpoint distance (TR RMa, SMa) [W-41] | 0 (3D breakpoint distance) |

Writer comments: R3a adds these rows to `qd_conf_pl_comments` in `src/qd_scenpar.cpp` (before the last row):

```cpp
    {"hE_rand", "", "1 = random environment height (TR 38.901 Table 7.4.1-1, note 1); 0 = fixed hE"},
    {"LOSmax", "", "0 = no maximum with the LOS-PL; 1 = PL = max( PL_LOS, PLn )"},
    {"Ax", "", "TX-RX 3D dist.-dep. of the additional PL term in the maximum [dB/log10(m)]"},
    {"Bx", "", "Reference PL of the additional term in [dB] @ 1 GHz, 1 m TX-RX dist."},
    {"Cx", "", "Freq.-dep. of the additional PL term in [dB/log10(GHz)]"},
    {"BP2D", "", "1 = PL after the break-point from the 2D break-point distance; 0 = from the 3D distance"},
```

Tests of the writer rows: the case "path-loss comments and formulas depend on the model" of
`test_conf_file_write.cpp` gets one line per new key (6 keys) checking its comment [R1-04].

The formula blocks `qd_conf_pl_formulas` are not changed (the written files of existing tests stay the same).

### 4.4 Values in the 23 v19 files

Lines are written in the style of the files (key name, `=`, value aligned with the neighbouring lines, `%` comment).
The names below are the scenario names without the prefix `3GPP_38.901_v19_`.

#### 4.4.1 R1: new numeric keys and the InF delay spread [W-33, I-3]

| File | Section of the file | New / changed lines |
|---|---|---|
| RMa_NLOS | "Decorrelation distance for the small-scale fading spatial consistency", after `SC_lambda` | `LOS_state_lambda = 60` |
| UMi_NLOS, UMa_NLOS, SMa_NLOS | same | `LOS_state_lambda = 50` |
| Indoor_NLOS | same | `LOS_state_lambda = 10` |
| InF_NLOS_SL, InF_NLOS_SH | same | `LOS_state_lambda = 5` (d_clutter / 2) |
| InF_NLOS_DL, InF_NLOS_DH | same | `LOS_state_lambda = 1` (d_clutter / 2) |
| RMa_LOS_O2I, RMa_NLOS_O2I | same section | `IO_state_lambda = 50` |
| UMi_*_O2I, UMa_*_O2I, SMa_*_O2I (6 files) | same section | `IO_state_lambda = 50` |
| RMa_LOS_O2I, RMa_NLOS_O2I | "Model parameters", last paragraph | `O2I_d2D_in_max = 10` |
| UMi_*_O2I, UMa_*_O2I, SMa_*_O2I (6 files) | "Model parameters", last paragraph | `O2I_d2D_in_max = 25` (SMa: commercial buildings; residential 10 m) |
| InF_LOS | "Large scale distributions", DS paragraph | `DS_mu = -9.35`, `DS_VS = 4.2727272727272725`, `DS_VS_a = 26`, `DS_VS_b = 14` [W-13] |
| InF_NLOS_SL, InF_NLOS_DH | same | `DS_mu = -9.44`, `DS_VS = 4`, `DS_VS_a = 30`, `DS_VS_b = 32` |
| InF_NLOS_DL, InF_NLOS_SH | same | `DS_mu = -9.44`, `DS_VS = 4.545454545454546`, `DS_VS_a = 30`, `DS_VS_b = 32` |

Comments:
- LOS-state lines: `% decorrelation distance of the LOS/NLOS state [m], Table 7.6.3.1-2; used by the caller`.
  The section header line `% See: 3GPP TR 38.901 v19.5.0; Table 7.6.3.1-2` stays.
- IO-state lines: `% decorrelation distance of the indoor/outdoor state [m], Table 7.6.3.1-2; used by the caller`.
- `O2I_d2D_in_max`: `% maximum indoor 2D distance [m], clause 7.4.3.1; argument d2d_in_max of o2i_loss`.
- InF DS: replace the summary line `% DS  = -7.2535 + Xds * ( 0.15 )` by
  `% DS  = log10( 26 * V/S + 14 ) - 9.35 + Xds * ( 0.15 )` (NLOS: `log10( 30 * V/S + 32 ) - 9.44 ... ( 0.19 )`) and
  the three comment lines after `DS_sigma` by:
  ```
  % The TR gives the InF delay spread as a function of the hall volume V and surface area S (Table 7.5-6 Part-3):
  % mu_lgDS = DS_mu + log10( DS_VS_a * V/S + DS_VS_b ). DS_VS is the V/S of the calibration hall of TR 7.8.4
  % (120 x 60 x 10 m: 4; 300 x 150 x 10 m: 50/11); lsp_generate takes another hall as argument vs_ratio.
  ```
  InF_LOS: `% (LOS: the average of the two calibration halls, 47/11.)` as fourth line.

#### 4.4.2 R3a: path-loss keys and constants [W-12, W-41, I-5]

| Files | Change |
|---|---|
| Indoor_NLOS, InF_NLOS_DH, _DL, _SH, _SL, UMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I, UMi_LOS, _LOS_O2I, _NLOS, _NLOS_O2I (13 files) | `PL_E = 13.333333333333334` (was 13.34); comment `E = 4 / 0.3 = 40/3` |
| RMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I, SMa_LOS, _LOS_O2I | `PL_E = 20.943951023931955` (was 20.944); comment `E = 2 pi / 0.3 = 20 pi / 3` |
| RMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I | `PL_A1 = 20.477913900145722`, `PL_B = 31.74083179916828`, `PL_D = 0.0013979400086720378` |
| RMa_NLOS, _NLOS_O2I | `PL_Bn = 31.754962063305896` |
| SMa_LOS, _LOS_O2I | `PL_A1 = 21.57442238074932`, `PL_B = 30.132619360949676` (`PL_D = 0.002` exact) |
| SMa_NLOS, _NLOS_O2I | remove `PL_A1`, `PL_A2`, `PL_B`, `PL_C`, `PL_E`, `PL_hE`; add `PL_LOSmax = 0` |
| UMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I | add `PL_hE_rand = 1` after `PL_hE` |
| RMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I, SMa_LOS, _LOS_O2I | add `PL_BP2D = 1` after `PL_hE` [W-41] |
| InF_NLOS_DL | add `PL_Ax = 25.5`, `PL_Bx = 33`, `PL_Cx = 20` |

Comment changes in the path-loss sections:
- UMa: `hE = 1 m (note 1; the random hE for hUT > 13 m is not modelled)` becomes
  `hE = 1 m or random for hUT >= 13 m (note 1; PL_hE_rand = 1, drawn from column 10 of lsf_draw)`;
  `E = 4 / 0.3 = 13.34 (rounded)` becomes `E = 4 / 0.3 = 40/3`.
- RMa, SMa: `E = 2 pi / 0.3 = 20.944` becomes `E = 2 pi / 0.3 = 20 pi / 3`; a line
  `% The constants are evaluated with full double precision for h and W given above.`; and (6 files with
  `PL_BP2D`) `% PL2 = PL1( dBP ) + 40 log10( d3D / dBP ) with the 2D breakpoint distance dBP (PL_BP2D = 1).`
- SMa NLOS: replace the two lines starting `% The TR has no max( PL_LOS, PLn ) for SMa NLOS.` by
  `% The TR has no max( PL_LOS, PLn ) for SMa NLOS: PL_LOSmax = 0 switches the LOS term off.`; the QuaDRiGa
  dual-slope formula block is removed from this file.
- InF_NLOS_DL: replace `% The TR takes the maximum of three terms ... below about 26 m.` by
  `% The TR takes the maximum of three terms for InF-DL: max( PL, PL_LOS, PL_InF-SL ). The InF-SL term is the
  % additional term PLx = Ax * log10( d3D ) + Bx + Cx * log10( fGHz ) of the QuaDRiGa model "nlos".`
- D27, in UMa_NLOS, UMi_NLOS, Indoor_NLOS after the TR formula lines:
  ```
  % Optional NLOS formula of Table 7.4.1-1 (UMa): PL = 32.4 + 20 log10(fc) + 30 log10(d3D), SF std 7.8 dB.
  % It is used with PL_model = logdist, PL_A = 30, PL_B = 32.4, PL_C = 20, PL_SF = 7.8.
  ```
  UMi: `31.9 log10(d3D)`, SF 8.2 (`PL_A = 31.9`, `PL_SF = 8.2`); Indoor: `PL' = 32.4 + 20 log10(fc) + 31.9
  log10(d3D)`, SF 8.29, used in `max(PL_LOS, PL')` (model `nlos` with `PL_An = 31.9`, `PL_Bn = 32.4`, `PL_Cn = 20`,
  `PL_sig = 8.29`).

After R3a every v19 file reads without ignored keys, passes the value checks and keeps a positive definite
correlation matrix (existing test).


## 5. Catch2 tests

### 5.0 Conventions for all new test files

- One file per function: `tests/catch2_tests/test_lsf_init_phases.cpp`, `test_lsf_draw.cpp`, `test_path_loss.cpp`,
  `test_lsp_generate.cpp`, `test_los_probability.cpp`, `test_o2i_loss.cpp`. No test of the internal header: every
  helper is covered through a public function with an independent oracle (5.7).
- Header comment as SPEC_CPP_API (what is pinned, oracles, conventions). Include `"quadriga_gbsm.hpp"` (and
  `"quadriga_sos.hpp"` where SOS functions provide supporting data).
- Diagnostics [W-24]: `#include "qd_test_diag.hpp"` and `QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")`; R2a adds the line
  `//   QD_DIAG_GBSM   large-scale model tests (test_lsf_*.cpp, test_path_loss.cpp, test_lsp_generate.cpp, ...)` to
  the list in `tests/catch2_tests/qd_test_diag.hpp`. Tables in CSV form with `# diag:` title.
- Test case names `"<function> - <statement>"`, unique in the suite; one behaviour per case; one validation case
  per function named `"<function> - input validation"` with a valid baseline (`CHECK_NOTHROW`) and one `SECTION`
  per check of the validation list.
- Oracles are typed into the test file: TR 38.901 v19.5.0 formulas and tables (cited by clause, table and equation
  number), the formulas of this spec (written out, not cited by section), a scalar sum-of-sinusoids
  `ref_sos(F, phi, pos) = sqrt(2/N) · sum_n cos(2·pi·(F(n,0)·x + F(n,1)·y + F(n,2)·z) + phi(n))`, the uniform map
  `0.5·erfc(-v/sqrt(2))`, a test-side linear ACF interpolation, a test-side 8x8 Cholesky. Supporting data from the
  library: `sos_preset` (frequency and ACF tables), `sos_init_phases`, `conf_file_read` (the files under test),
  `lsf_init_phases` / `lsf_draw` where a test of another function needs phases or random values.
- Never an oracle: the function under test or `quadriga_gbsm_helpers.hpp` (except in the internal test).
- Random inputs of statistical tests come from `std::mt19937_64` with fixed seeds and test-side transforms
  (Box-Muller for Normal values); `std::normal_distribution` is not used (not portable).
- Tolerances: exact-by-construction double results 1e-12 (absolute for O(1) values, relative for others); dB values
  1e-9 where long chains of `lg` and constants enter; statistical tolerances = 5 standard errors plus the model
  approximation error, stated in a comment.
- Tolerances of values produced by `lsf_draw` (AVX2 kernel with single-precision sin/cos, 3.0.1) [W-44], with the
  reason in a comment: Normal columns 2e-5 absolute for 256 sinusoids, scaled with `sqrt(n_coeff / 256)` for other
  presets (1.5e-5 for 128, 4e-5 for 1024); 3e-5 for the coupled columns 3 to 6 in dual mobility (two kernel
  errors, coupling gain up to sqrt(2)); Uniform columns 1e-5 (the slope of the uniform map is at most 0.4); zero
  sample means through `lsf_draw` 2e-5. Checks on phases with a test-side `ref_sos` (no `lsf_draw` involved) stay
  1e-12. Equality checks that compare the same evaluation stay exact: z independence in 2D mode, independence of
  the other links, determinism, swap reciprocity, `floor_id` all 0 equals `nullptr`, single mobility ignores
  `tx_pos`. `lsp_generate`, `path_loss`, `los_probability`, `o2i_loss` take typed random values and keep the
  double tolerances above.
- Test files contain no internal IDs (D-, W-, I-, R1-), no section numbers of this spec, no `md/` or local paths and
  no QuaDRiGa-NG line numbers (9.1); comments cite TR 38.901 clauses, tables and equations or state the rule.
- Determinism: a repeated call gives identical results (one check per file).

### 5.1 `test_lsf_init_phases.cpp`

Scenpar for most cases: all 10 decorrelation distances set explicitly (e.g. DS 11, KF 12, SF 13, ASD 14, ASA 15,
ESD 16, ESA 17, XPR 18, absTOA 19, SC 20 m) so that a mix-up of generators is visible; every `ref_sos` of
generator g uses `sos_preset(preset, F_g, acf_g, dist_g, lambda_g)` with that generator's own distance.

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| output size follows the preset | `[256, 11, 2]` for Comb256, `[128, 11, 2]` for Exp128, `[1024, 11, 2]` for Gauss1024 | preset names | exact |
| slice 0 equals sos_init_phases with the same seed | `phase.slice(0) == sos_init_phases(n_coeff, 11, seed)` (no positions) | documented contract | exact |
| slice 1 is the mirror of slice 0 | `slice1.col(g) == slice0.col(m(g))` for all g, with and without bias removal, single and dual mobility | mirror table (ASD <-> ASA, ESD <-> ESA, others themselves) | exact |
| the same seed gives the same phases | two calls seed 7 equal; seed 8 differs | — | exact |
| bias removal zeroes the drop mean in single mobility | 200 Rx positions in a 100 m x 100 m square (z from 1.5 to 30 m); mean over the positions of `ref_sos(F_g, slice0.col(g), (x, y, 0))` is 0 for g = 0..8 (also for ASD/ASA and ESD/ESA with their different distances); columns 9, 10 equal those of the same seed without bias removal | `ref_sos` | 1e-12 (closed-form correction; rounding of 200 O(1) values) |
| bias removal in 3D mode uses the z coordinate | `horizontal = false`: mean of `ref_sos(.., (x, y, z))` is 0; the mean of the 2D evaluation is not (> 1e-6 for at least one g) | `ref_sos` | 1e-12 |
| bias removal with dual mobility uses the Rx and Tx positions together | 100 Rx positions as above, Tx = Rx + (30, -20, 5) for every link (equal separation): mean over the 200 positions `[R; T]` (z = 0) of `ref_sos(F_g, slice0.col(g), .)` is 0 for g = 0..8; the link mean of `lsf_draw(..., dual_mobility = true)` columns 0, 1, 2, 7, 8 is 0; the phases from swapped `rx_pos`/`tx_pos` equal those of the original call modulo 2·pi (the union is summed in another order) | `ref_sos`; `lsf_draw` as supporting function (tested in 5.2) | 1e-12 (`ref_sos`; phases compared by `abs(remainder(a - b, 2·pi))`); 2e-5 (`lsf_draw` mean) |
| no bias removal for fewer than 3 positions, without positions or when switched off | phases equal the raw phases in all three cases; `rx_pos` with 0 rows does not throw | `sos_init_phases` + mirror | exact |
| a decorrelation distance of 0 is used as 0.1 m | `DS_lambda = 0`, bias removal: mean of `ref_sos` with `sos_preset(.., 0.1)` is 0 | `ref_sos` | 1e-12 |
| input validation | one section per check of 3.1: unknown preset (`Input 'preset' is not a known preset.`); `DS_lambda` -1 and `SC_lambda` Inf (the `scenpar_validate` message passes through); an unknown key; `rx_pos` 2 columns; `rx_pos` NaN; `tx_pos` 2 columns; `tx_pos` 2 rows for 5 links; dual mobility with positions, bias removal, no `tx_pos`; baseline: the same call without bias removal does not throw | — | — |

### 5.2 `test_lsf_draw.cpp`

Tolerances of `lsf_draw` values follow 5.0 (AVX2 kernel); the reason is stated in a comment of each case.

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| single mobility values equal the sum of sinusoids of each generator | 7 links with distinct coordinates (negative, up to 300 m, z != 0), distinct lambdas (as in 5.1); column g equals `ref_sos(F_g, slice0.col(g), (x, y, 0))`, columns 9, 10 the uniform map of it; column 10 uses `SF_lambda` | `ref_sos`, `sos_preset(.., lambda_g)` | 2e-5 (Normal), 1e-5 (Uniform) |
| 3D mode uses the z coordinate | `horizontal = false`: `ref_sos(.., (x, y, z))` | as above | 2e-5 / 1e-5 |
| other presets | Exp128 and Gauss1024: column 0 equals `ref_sos` | as above | 1.5e-5, 4e-5 |
| 2D mode values do not depend on z | changing all z leaves `rnd` unchanged | — | exact |
| single mobility ignores tx_pos | two different `tx_pos` give identical `rnd` | — | exact |
| a link does not depend on the other links | links evaluated one by one equal the joint call | — | exact |
| dual mobility values follow the two-device formulas and the coupling | 6 links incl. one with Tx = Rx and one with distance > 5 lambda; self-mirrored columns `(ref_sos(F_g, P0_g, R) + ref_sos(F_g, P0_g, T)) / sqrt(2 + 2·A_g(D))`; columns 3..6 `(ref_sos(F_g, P0_g, R) + ref_sos(F_m, P0_m, T)) / sqrt(2)`, then the symmetric 2x2 root with `c = min(0.5·(A_g(D) + A_m(D)), 1 - 1e-5)`; 9, 10 the uniform map | typed formulas, `ref_sos`, test-side linear interpolation of the `sos_preset` ACF table | 2e-5 (0, 1, 2, 7, 8), 3e-5 (3..6), 1e-5 (9, 10) |
| swapping Tx and Rx swaps ASD with ASA and ESD with ESA | dual mobility, unequal lambdas of ASD/ASA and ESD/ESA, phases with and without bias removal: `rnd(tx=A, rx=B)` columns 0, 1, 2, 7, 8, 9, 10 equal `rnd(tx=B, rx=A)`, 3 <-> 4 and 5 <-> 6 swapped | reciprocity (same evaluations in swapped roles) | exact |
| floor 0 equals no floor ID and equal IDs share one field | `floor_id = 0` gives the values of `nullptr` (exact); links of floor 3 and floor 259 equal `ref_sos` with the phases plus the offsets `2·pi·k·n/N + psi_b(n)` typed in the test (`k = q mod N`, `b = q div N`; `psi_1(n) = 2·pi·((x_n >> 11)·2^-53)` with `x_n` the n-th output of `std::mt19937_64 gen(1)`) | typed offset rule | exact; 2e-5 |
| fields of different floors are uncorrelated at equal positions | lambda 10 m, 100 x 100 grid with 50 m spacing (n = 10^4, ACF at 5 lambda below 0.02), the same positions on every floor: sample correlation of every Normal column between floors (0, 1), (1, 5), (0, 255) below 0.05 in magnitude; between (0, 256) below 0.25 | zero within a block of N floor IDs, `1/sqrt(2N) = 0.044` between blocks | 0.05 = 5 standard errors of 1/sqrt(n); 0.25 = 5.6 · 0.044 |
| marginal distributions | same grid, floor 0: columns 0..8 `abs(mean) < 0.05`, `abs(std - 1) < 0.05`; columns 9, 10 in [0, 1], `abs(mean - 0.5) < 0.015`, fraction below 0.1 within 0.015 of 0.1 | N(0,1), U(0,1) | 5 standard errors (0.01, 0.003, 0.003) |
| the correlation follows the preset ACF | lambda 10 m, base grid as above, offsets 5, 10, 20 m in 4 directions: sample correlation within 0.08 of the `sos_preset` ACF at that distance (Comb256 and Exp256) | preset ACF table | 0.08 = 5 standard errors (0.05) + SOS approximation error (0.03) |
| sizes, empty input and pre-sized output | `n_link = 0` gives `[0, 11]` without an exception (single and dual mobility); a pre-sized `[n_link, 11]` output with stale content is overwritten; a wrong size is resized | — | exact |
| input validation | sections: `phase` `[128, 11, 2]` with Comb256, `[256, 10, 2]`, `[256, 11, 1]`, NaN entry; unknown preset (`Input 'preset' is not a known preset.`); `rx_pos` 2 columns; Inf in `tx_pos`; `tx_pos` 2 rows for 3 links; `floor_id` length; negative `XPR_lambda` (`scenpar_validate` message) | — | — |

### 5.3 `test_path_loss.cpp`

TR oracle: Table 7.4.1-1 typed with `c = 3e8`, RMa `h = 5, W = 20`, SMa `h = 10, W = 10`, `hE = 1` for UMa/UMi
(note 1 with C = 0), RMa/SMa LOS `PL2 = PL1(dBP) + 40·lg(d3D/dBP)` with the 2D breakpoint distance (the files set
`PL_BP2D = 1`), InF-DL `max(PL, PL_LOS, PL_InF-SL)`, SMa NLOS without max, the shadow-fading std of the table
(RMa/SMa LOS 4 or 6 by the side of `dBP`). The SF oracle of the O2I files is the O2I shadow fading of Table 7.5-6
(7 dB UMa/UMi, 8 dB RMa/SMa, equal to `SF_sigma` of the files; they have no `sig` keys), not the outdoor value of
Table 7.4.1-1 [R1-21]. Points are chosen at `0.5·dBP` and `2·dBP` (never at the breakpoint itself).

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| the v19 files reproduce TR 38.901 Table 7.4.1-1 | for every LOS, NLOS and O2I file: `pl` and `sf_sigma` at the points: UMa hBS 25, UMi hBS 10: hUT {1.5, 12}, fc {3.5, 28} GHz, d2D {15, 0.5 dBP', 2 dBP' if <= 5 km, 1000, 4500}; RMa hBS {35, 25}, hUT {1.5, 5}, fc {0.9, 3.5, 6}, d2D {50, 0.5 dBP, 2 dBP (LOS <= 10 km, NLOS <= 5 km), 4000}; SMa hBS {25, 35}, hUT {1.5, 4.5, 13.5}, fc {3.5, 20}, d2D {50, 0.5 dBP, 2 dBP, 3000}; InH hBS 3, hUT 1, fc {3.5, 28, 60}, d2D {1, 5, 20, 100}; InF hBS {1.5, 8}, hUT 1.5, fc {3.5, 28}, d2D {1, 5, 10, 30, 100, 500}. O2I files equal the outdoor TR formula at the geometric distance, `sf_sigma` the O2I value. A diagnostic counts the InF-DL points where the InF-SL term wins (> 0 asserted) | typed TR formulas | 1e-9 dB (full-precision constants in the files); `sf_sigma` exact |
| O2I distances: geometric distance without d3d_in, outdoor distance with it | UMa NLOS file: the call without `d3d_in` equals the TR formula at the geometric distance (covered above); with `d3d_in = 12` it equals the typed formula at `d3 - 12`, `d2·(1 - 12/d3)`; `d3d_in > d3D` clamps to 0.1 m; `d3D = 0` with `d3d_in` gives an indoor fraction 0 | typed legacy distance rule | 1e-12 dB |
| UMa random hE follows the probability of TR note 1 | UMa LOS file, fc 0.05 GHz, hBS 25, hUT 20, d2D 400 m, K = 10000 links with `rnd_he = (k + 0.5)/K`: hE of each link identified by matching `pl` with the TR LOS formula for hE in {1, 12, 15, 18} (the candidates differ by > 0.01 dB, asserted first); count of hE = 1 equals `K/(1 + C)` within 1; counts of 12, 15, 18 equal `K·(1 - p1)/3` within 2; hUT 13.2 and hUT 10: all hE = 1; `rnd_he` omitted: hE = 1; NLOS file with the same `rnd_he` uses the same hE in its LOS term | TR note 1 typed | match 1e-9 dB; counts deterministic |
| legacy models match the get_pl formulas | one SECTION per model with distinct parameter values, 4 links x 3 frequencies: constant; logdist with and without `SF`; logdist_simple; winner_los (links with different heights incl. < 1.5 m, both branches, breakpoint from each link's own heights, `C1 != C2` so that the second branch shows `C2`, continuity at the breakpoint for `F1 = F2`); winner_nlos (fc 1.0, 1.7, 2.5 GHz); winner_pathloss with `SF`; dual_slope (defaults A2, C, D; hE clamp 1.1·hE; both sides; sig1/sig2; `BP2D` absent and 1 with `hBS != hUT`, the two forms differ by the typed amount); tripple_slope (three regions, `E2` default, dBP1 > dBP2 case); nlos (all 13 terms non-zero, G2n default, `LOSmax` 0 and 1, `Ax/Bx/Cx`, `sig`, `BP2D` in the LOS term); satellite (`usePLa` 1 at 1, 20.71, 33 and 150 GHz, `usePLa` 0, 600 km height, below the horizon gives +Inf) | model formulas typed in the test; zenith-attenuation table typed | 1e-10 dB |
| distances below 0.1 m are clamped | co-located Tx and Rx give the loss at 0.1 m | typed | 1e-12 dB |
| default sf_sigma follows the LSP formula | `SF_sigma 6, SF_delta 1.5, SF_omega 0.5, SF_kappa 0.3, SF_tau -0.2, SF_beta 0.1, LSP_fc_min 6`, fc 3 and 10 GHz, logdist model without `SF` | LSP std formula typed | 1e-12 |
| an empty model gives zero loss | `pl == 0`, `sf_sigma` default | — | exact |
| optional outputs and pre-sized outputs | `pl` with and without `sf_sigma` equal; pre-sized outputs overwritten; `n_link = 0`; one `tx_pos` row equals repeated rows; repeated call identical | — | exact |
| input validation | sections for every check of 3.3.1 (unknown model; a missing required key for each model with required keys; empty value vector and a vector with two values (`Path-loss parameter 'B' must have one value.`); `E = 0`; `winner_los` `A1 == A2`; `d3d_in` negative and wrong length; `rnd_he` 1.5 and wrong length; `fc` empty, 0, NaN; positions; an invalid scenpar value passes the `scenpar_validate` message through) | — | — |

### 5.4 `test_lsp_generate.cpp`

The formula mechanics are pinned with custom scenpar values in which every term is non-zero and distinct (all
`_gamma`, `_epsilon`, `_zeta`, `_alpha`, `_delta`, `_kappa`, `_tau`, `_beta` terms, `_omega`, `LSP_fc_min`, the ESD
terms and floor, the V/S keys), typed against the mean/std formula written out in the test. The per-file check of
the 23 v19 files against Tables 7.5-6 to 7.5-12 stays in `test_conf_files_v19.cpp` (key level); this file adds TR
spot checks through `lsp_generate` for four files [R1-19]. Reference geometry: `tx = (0, 0, hBS)`,
`rx = (d2D, 0, hUT)`. Where the TR has "N/A" (KF of NLOS and O2I links) the oracle is the QuaDRiGa convention of
the files: `KF_mu = -100` dB, `KF_sigma = 0`.

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| mean and std follow the formula with every term | custom keys as above, 3 links (hUT above and below hBS, a point where the ESD floor applies), fc {1.5, 3.5, 28} GHz with `LSP_fc_min = 2`: `lsp_mu` and `lsp_sigma` of all 8 slices; SF mean 0 | mean/std formula typed | 1e-12 |
| TR spot checks of four v19 files | `lsp_mu`, `lsp_sigma` of all 8 slices: UMa NLOS (hBS 25, hUT 1.5 and 12, d2D 200 m, fc 3.5 and 28 GHz: the 6 GHz clamp); UMi LOS (hBS 10, fc 1.5 and 28: the 2 GHz clamp); SMa NLOS (hBS 35, hUT 1.5 and 4.5: the ESD height terms); InF NLOS DL (hBS 1.5, hUT 1.5, d2D 30 m, DS with the file default V/S, `vs_ratio` 4 and 50/11) | Table 7.5-6 Parts 1, 3, 4, Tables 7.5-7, 7.5-8, 7.5-12, typed | 1e-12 |
| EoD offset formula and sign | custom `EoD_offset_*` keys with every term active (`b` > 0 and 0, `hcl` above and below hUT), d2D {5, 50, 500, 2000}; spot checks against the TR: UMa NLOS (6 GHz clamp) and SMa NLOS (hUT above and below 3.5 m) give `-mu_offset,ZOD` of Tables 7.5-7 and 7.5-12; UMa LOS gives 0 | EoD formula typed; TR tables typed | 1e-10 deg |
| values are 10^(mu + sigma x) with x from the correlation root | identity correlation, distinct mu/sigma per LSP, given `rnd`: `lsp = 10^(mu + sigma·s)` (KF, SF, XPR: `10^(0.1·(...))`) | typed | 1e-12 relative |
| the Cholesky root follows the TR order | a custom positive definite matrix with all 28 correlations non-zero, mu = 0, sigma = 1 (dB for KF, SF, XPR): links with `s = e_j` give column j of `Q` built by the test-side Cholesky of the permuted matrix; `x_SF = s_SF`; `Q·Q^T = R` | test-side Cholesky | 1e-12 |
| the legacy root is the symmetric square root [pending user decision R1-09] | `cholesky = false`, links with `s = e_j` give the columns of `X`: `X` symmetric, `X·X = R`, `X` positive definite (test-side Cholesky of `X` succeeds); also for a matrix with one correlation of 0.999999 (smallest eigenvalue about 1e-6) and for the identity (X = I) | uniqueness of the SPD root | 1e-12 |
| sample covariance of the log values equals the correlation matrix | 20000 links with Box-Muller `rnd`, UMa NLOS file correlations, both roots [pending user decision R1-09: Cholesky only]: sample covariance of `x` within 0.04 of `R` | R | 0.04 = 5 standard errors (<= 0.008) |
| spreads are clipped at 104 and 52 degrees | mu of ASD, ASA 2.5 and of ESD, ESA 2.0: clipped with `clip_spreads = true`, not with false; `lsp_mu` the same in both | — | exact |
| the same random values apply to all frequencies | `(lg DS - mu)/sigma` equal at 3 frequencies | TR 7.6.5 | 1e-12 |
| sf_sigma from path_loss replaces the SF standard deviation | `lsp SF = 10^(0.1·sf_sigma·x_SF)`, `lsp_sigma` slice 2 = `sf_sigma`, other slices unchanged | typed | 1e-12 |
| LSP_fc_min clamps the frequency | with `LSP_fc_min = 6`: `lsp_mu`, `lsp_sigma` at 2 GHz equal those at 6 GHz, differ at 10 GHz | — | exact / > 1e-6 |
| InF delay spread uses V/S | InF LOS file: default (47/11), `vs_ratio` 4 and 50/11; `vs_ratio` overrides `DS_VS`; `DS_VS = 0` and `vs_ratio = 0` give no V/S term | Table 7.5-6 Part-3 typed | 1e-12 |
| absolute TOA | `absTOA_mu = -30` gives 0, -29.99 active; value `10^(mu + sigma·rnd(:,8))`; `absTOA_Lmax` bound on both sides of `2L/c` | rule typed (active above -30, bound `2·L/c`, `c = 299792458`) | 1e-12 relative |
| ground permittivity interpolates dry, medium and wet ground | U = 0, 0.25, 0.5, 0.75, 1 at f 1, 10, 28 GHz; imaginary part >= 0 | permittivity model typed | 1e-12 |
| optional outputs alone and together give the same values | every optional output alone vs. all together; pre-sized outputs; `n_link = 0`; one `tx_pos` row vs repeated | — | exact |
| input validation | sections for every check of 3.4.1 (incl. `rnd` 10 columns, NaN in `rnd`, `sf_sigma` size, `sf_sigma` -1 and NaN, `vs_ratio` -1, `vs_ratio > 0` with a = b = 0, `DS_VS > 0` with a = b = 0 and `LSP_fc_min` -1 (both `scenpar_validate` messages), a not positive definite matrix, column 9 = 1.5 only when `gr_eps_re` is requested) | — | — |

### 5.5 `test_los_probability.cpp`

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| TR 38.901 models match Table 7.4.2-1 | each model on both sides of every breakpoint: RMa d {5, 10, 10.001, 500, 5000}; UMi {0, 18, 18.5, 100, 1000}; UMa hU {1.5, 15, 22.5, 25} x d {10, 18, 30, 100, 400, 1000}; InH Mixed {1, 1.2, 3, 6.5, 6.6, 50}; Open {3, 5, 20, 49, 50, 200}; InF SL/DL/SH/DH with defaults and with explicit arguments, hB 8, hU 1.5, d {1, 5, 20, 50}; HH; SMa with r_veg {0, 0.1, 0.2}, hU {1.5, 4.5, 13.5}, d {5, 10, 11, 100, 1000} | typed TR, limited to [0, 1] | 1e-14 |
| 38.811, QuaDRiGa and fixed models | 38.811 tables at elevations 5, 10, 45, 87, 90 deg and below the horizon; Industrial with hB 4, 8, 12; LOS, NLOS; the TR 36.873 UMi formula `min(18/d, 1)·(1 - exp(-d/36)) + exp(-d/36)` equals `38.901_UMi` at d {5, 18, 30, 300} | typed (elevation tables, QuaDRiGa formula, TR 36.873 formula) | 1e-14 |
| probabilities stay in [0, 1] | UMa at d 20 m, hU 20 m gives 1 (TR formula > 1); every model over a grid of d and heights in [0, 1] | limit to [0, 1] | exact |
| InF edge rules | hU >= h_c gives 1; hB <= hU < h_c gives the SL rule; r = 0 gives 1 | edge rules typed | 1e-14 |
| SMa edge rules | hB <= hU gives 1; hU 18 m (above the vegetation height of 15 m) sets the vegetation term to 0 | edge rules typed | 1e-14 |
| soft LOS is 0.5 at G = -F(d) and matches p over many G | `rnd_g = -F(p)` with F from a test-side bisection on `std::erfc` (to 1e-15) gives 0.5; quantile grid `G_k = F((k + 0.5)/K)`, K = 10000: fraction of `los_soft > 0.5` equals p within 1/K; p = 1 gives 1, p = 0 gives 0; eq. (7.6-18) typed at fc 3.5 and 28 GHz | test-side probit, eq. (7.6-18) | `k·1e-10·max(1, abs(F))/pi + 1e-15` (from the probit target, `k = sqrt(20/lambda)`); counts within 1/K |
| the inverse normal CDF of the soft LOS is accurate | `38.901_RMa` with `d = 10 - 1000·ln(p_target)` for p on a logarithmic grid in [1e-12, 0.5] and `d` just above 10 m for p up to 1 - 1e-12; `rnd_g = 0`, fc = 1 kHz (`k = sqrt(20/lambda)` about 8e-3, so `atan` is nearly linear): `F_rec = tan(pi·(los_soft - 0.5)) / k` equals the test-side bisection probit of the returned `p_los` | bisection on `std::erfc` | `1e-10·max(1, abs(F))` |
| optional inputs | `los_soft` without `rnd_g` throws; `rnd_g` without `los_soft` is ignored; scalar `h_bs` equals the repeated vector | — | exact |
| input validation | sections for every check of 3.5.1 | — | — |

### 5.6 `test_o2i_loss.cpp`

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| wall and indoor losses follow Tables 7.4.3-1 and 7.4.3-2 | one-hot weights for `low`, `high`, `low_A`, `high_v14` [pending user decision R1-10: without `high_v14`]; fixed `rnd` per UT; f 3.5, 28, 60 GHz: `loss = PL_tw + 0.5·d2din + sigma_P·N` | typed tables | 1e-12 dB |
| the backward compatible model of Table 7.4.3-3 | `legacy_6GHz`: `20 + 0.5·U1·d2d_in_max`, independent of column 3 | typed | 1e-12 |
| car penetration loss | `car`: `9 + 5·N`, `car_metal`: `20 + 5·N`, `d2d_in = 0`, independent of f | 7.4.3.2 typed | 1e-12 |
| mmMAGIC indoor loss and elevation term [pending user decision R1-10] | `alpha`, `d_room` from columns 4, 5; `loss_el = 20·abs(theta)/90` with theta from hand-built geometry (UT above and below the BS); 0 for other models | mmMAGIC model typed | 1e-12 |
| the model is chosen by the cumulative weights | 1000 UTs with `rnd(:,0) = (k + 0.5)/1000`, weights {0.2, 0, 0.5, 0.3}: counts 200, 0, 500, 300; `u = 1` picks the last non-zero weight; `u = 0` never picks a zero weight | cumulative-weight rule | exact |
| indoor distance is the minimum of two uniforms | `d2d_in = min(U1, U2)·d_max`; grid 100 x 100 of (U1, U2): fraction with `d2d_in <= x·d_max` equals `1 - (1 - x)²` within 0.02; legacy [pending user decision R1-10]: at the table CDF values the table distances, between them linear | TR 7.4.3.1; legacy table typed | 1e-12; grid 2/100 |
| indoor part of the 3D distance [pending user decision R1-10: moves to `path_loss`] | hand-built UT/BS pairs: `d3 · din / d2`; `din > d2` clamp; outdoor distance below 10 m rule; `d2 = 0`; car gives 0 | geometry rule typed | 1e-12 |
| optional outputs and sizes | each optional output alone vs together; pre-sized; `n_ut = 0`; 4-column and 6-column `rnd` give the same result for non-mmMAGIC models | — | exact |
| input validation | sections for every check of 3.6.1 | — | — |

### 5.7 Internal helpers

No test file of its own [R1-08]: the internal header holds only helpers with public users, and each is covered
there with an independent oracle.

| Helper | Covered by |
|---|---|
| `qd_gbsm_lsf_generators`, `qd_gbsm_freq`, `qd_gbsm_positions` | 5.1, 5.2 (`ref_sos` with each generator's own distance, 2D/3D, Tx broadcast, unknown preset) |
| `qd_gbsm_lsp_mu_sigma` | 5.4 (formula with every term), 5.3 (default `sf_sigma`) |
| `qd_gbsm_chol8` | 5.4 (TR-order root against a test-side Cholesky, not positive definite), existing conf tests (positive-definite flag of `qd_scenpar_xcorr`) |
| floor offsets (file-local) | 5.2 (typed offset rule, zero correlation within a block) |
| legacy root (file-local) | 5.4 [pending user decision R1-09] |
| probit (file-local) | 5.5 (accuracy through `los_soft`) |

### 5.8 Changes to existing tests

- R1: `test_conf_file_read.cpp`, `test_scenpar_validate.cpp`, `test_conf_file_write.cpp` (165 -> 171 keys, 27 ->
  33 new keys, 166 -> 172 assignments with `SubpathMethod`, the key-order oracle extended by the 6 names, the new
  cross-key rule of 4.2 with a negative case in the validation test) [R1-04]: the 6 keys are added to the
  `QD_NONNEG` lists of the cases "value checks per validation class" in `test_conf_file_read.cpp` and
  `test_scenpar_validate.cpp`; the cross-key rule through `scenpar_validate`, through `conf_file_read` (message
  with file name and the line of `DS_VS`) and through `conf_file_write`; in `test_conf_file_write.cpp` a case "the
  new keys land in their section and paragraph and are not written at their default" (`LOS_state_lambda`,
  `IO_state_lambda` in the paragraph after `SC_lambda` of the small-scale section, `O2I_d2D_in_max` in its own
  paragraph of "Model parameters", the `DS_VS` keys in the DS paragraph; none of the 6 lines at value 0) and a
  case "the DS overview line shows the V/S term" next to the ESD overview case (`DS_VS > 0` gives
  `% DS  = log10( 26 * V/S + 14 ) - 9.35 + Xds * ( 0.15 )`, `DS_VS = 0` the line of today); `test_conf_files_v19.cpp`: include
  `quadriga_gbsm.hpp`, the InF DS checks (`DS_mu` -9.35 / -9.44, `DS_VS`, `DS_VS_a`, `DS_VS_b` and
  `DS_mu + lg(a·VS + b)` against the TR formula for the hall of 7.8.4), the values of the 6 keys per file (4.4.1),
  `LOS_state_lambda` and `IO_state_lambda` 0 in all other files; `test_conf_file_list.cpp`: include. MEX tests
  `test_conf_file_read.m`, `test_scenpar_validate.m`, `test_conf_file_write.m` (166 -> 172 fields, field indices of
  the new keys; the `DS_VS` cross-key error through `scenpar_validate`); Python tests of the four functions call
  `quadriga_lib.gbsm.*`, counts 166 -> 172, the `DS_VS` cross-key error (`ValueError`), plus one check that
  `quadriga_lib.tools` has no `conf_file_read` any more.
- R3a: `test_conf_files_v19.cpp` case "path-loss constants of RMa and SMa follow Table 7.4.1-1": tolerance 1e-12
  (full precision), SMa NLOS asserts `LOSmax == 0` and the absence of `A1, A2, B, C, E, hE`; new checks of
  `PL_hE_rand` (UMa files), `PL_BP2D = 1` (RMa LOS/NLOS and SMa LOS files incl. O2I) and `PL_Ax/Bx/Cx` (InF-DL);
  the comment "E and hE ... needed by the QuaDRiGa model" is removed. `test_conf_file_write.cpp` case "path-loss
  comments and formulas depend on the model": one check per new `PL_` key comment (`hE_rand`, `LOSmax`, `Ax`,
  `Bx`, `Cx`, `BP2D`) [R1-04].


## 6. Wrappers

General rules (both APIs): section "Geometry-based stochastic model"; one-line summaries identical to the C++ blocks;
thin wrappers, no numerical logic; `scenpar` is the struct / dict returned by `conf_file_read` or
`scenpar_validate` (the entry `SubpathMethod` is accepted and ignored), `[]` / `None` = all defaults; `plpar` is the
struct / dict of `conf_file_read` whose entry `model` is passed as `pl_model` (no `model` entry or `[]` / `None` =
empty model, zero loss) [W-25]. The pending decisions R1-09 and R1-10 change the Usage lines, argument counts
and test points exactly as listed in 2.1 and 3.6.0. Helpers: MEX `qd_mex_struct2scenpar`, `qd_mex_struct2plpar`
(`src/mex_quadriga_lib_functions.hpp`, include `mex_quadriga_lib_functions.hpp`); Python `qd_python_dict2scenpar`,
`qd_python_dict2plpar` (`api_python/python_quadriga_adapter.hpp`). `floor_id` is read as double, checked to be
non-negative integers (`Entries in 'floor_id' must be non-negative integers.`) and converted to `arma::uvec`
[W-31].

### 6.1 MEX (`api_mex/<name>.cpp`, `tests/mex_tests/test_<name>.m`)

| Function | Usage | Inputs (defaults) | Notes |
|---|---|---|---|
| lsf_init_phases | `phase = quadriga_lib.lsf_init_phases( scenpar, seed, rx_pos, tx_pos, dual_mobility, remove_bias, preset, horizontal );` | scenpar; seed -1 (int64); rx_pos `[]`; tx_pos `[]`; dual_mobility false; remove_bias true; preset 'Comb256'; horizontal true | nrhs 1..8, nlhs <= 1; `phase` `[n_coeff, 11, 2]` copied after the call (size from the preset) |
| lsf_draw | `rnd = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, preset, horizontal );` | floor_id `[]` (all on floor 0); rest as above | nrhs 4..8, nlhs <= 1; `rnd` `[n_link, 11]` via `qd_mex_init_output`; floor IDs are labels, no index shift |
| path_loss | `[ pl, sf_sigma ] = quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d3d_in, rnd_he );` | d3d_in `[]` (= geometric distance, TR); rnd_he `[]` | nrhs 5..7, nlhs <= 2; outputs `[n_link, n_freq]` pre-allocated |
| lsp_generate | `[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = quadriga_lib.lsp_generate( scenpar, fc, tx_pos, rx_pos, rnd, sf_sigma, vs_ratio, cholesky, clip_spreads );` | sf_sigma `[]`; vs_ratio 0; cholesky true [pending user decision R1-09]; clip_spreads true | nrhs 5..9, nlhs <= 7; all outputs pre-allocated (`[n_link, n_freq, 8]`, `[n_link, 1]`, `[n_link, n_freq]`) |
| los_probability | `[ p_los, los_soft ] = quadriga_lib.los_probability( model, d2d_out, h_bs, h_ut, d_clutter, r_clutter, h_clutter, rnd_g, fc );` | d_clutter, r_clutter, h_clutter NaN (= model default); rnd_g `[]`; fc `[]` | nrhs 4..9, nlhs <= 2; `los_soft` is `[]` when `rnd_g` or `fc` is empty, otherwise `[n_link, n_freq]` |
| o2i_loss [pending user decision R1-10] | `[ loss, d2d_in, model_index, d3d_in, loss_el ] = quadriga_lib.o2i_loss( fc, rnd, models, weights, d2d_in_max, tx_pos, rx_pos, d2d_in_min_of_two );` | d2d_in_max 25; tx_pos `[]`; rx_pos `[]`; d2d_in_min_of_two true | nrhs 4..8, nlhs <= 5; `models` cell array of char (or one char array); `model_index` uint64, **1-based** (shift +1 after the call); `d3d_in`, `loss_el` `[n_ut, n_tx]` or `[]` without positions |

Documentation blocks follow SPEC_MATLAB_API (UPPERCASE heading, `## Usage:`, 100-character lines). Each MEX test
covers the checklist of SPEC_MATLAB_API "What every test covers" with these specific points:

- lsf_init_phases: size `[256, 11, 2]`; same seed equal, `seed = []` default runs; `int64(7)` seed equals 7;
  positions as `single`; mirror relation of slice 2 (MATLAB index) without positions; bias removal changes slice 1
  for 3 positions or more; errors: argument counts, unknown preset (C++ message `Comb999`), dual mobility without
  `tx_pos` (C++).
- lsf_draw: `[n_link, 11]` double; z-independence in 2D, z-dependence with `horizontal = false`; `floor_id = []`
  equals all zeros; `floor_id` with `-1` and `1.5` raise the wrapper error; `uint8` floor IDs accepted; dual
  mobility swap of columns 4/5 (MATLAB 1-based) with swapped positions; `phase` of the wrong size raises the C++
  error.
- path_loss: logdist via a hand-made `plpar` struct (`model = 'logdist'`, `A = 20`, `B = 30`, `C = 20`) at d3D =
  100 m, 1 GHz gives 70 dB; `plpar = []` gives zeros; a v19 file read with `conf_file_read` runs; `sf_sigma`
  output with `SF` field; `d3d_in` given changes the result (legacy distance); `rnd_he` with the UMa
  file; error: unknown model (C++).
- lsp_generate: sizes of all 7 outputs; `rnd = zeros` gives `lsp = 10.^lsp_mu` for DS; `clip_spreads = false`
  allows ASD > 104; `cholesky = false` changes the KF/SF mixing [pending user decision R1-09]; `sf_sigma` replaces slice 3 (MATLAB index) of
  `lsp_sigma`; `vs_ratio` changes the InF DS; error: `rnd` with 10 columns (C++).
- los_probability: `38.901_UMi` at d = 10 m gives 1; `'NLOS'` gives 0; `h_bs` scalar vs vector; `los_soft` `[]`
  without `rnd_g`; `los_soft` `[n_link, n_freq]` with both; `d_clutter = []` equals NaN equals omitted; error:
  unknown model (C++).
- o2i_loss: `models = 'car'` (char) and `{'car'}` equal; `model_index` uint64 starting at 1; `d3d_in`, `loss_el`
  `[]` without positions, sizes `[n_ut, n_tx]` with them; mmMAGIC needs 6 columns (C++ error); weights not summing
  to 1 (C++ error).

### 6.2 Python (`api_python/qpy_<name>.cpp`, submodule `gbsm` [D29], `tests/python_tests/test_<name>.py`)

Dispatcher `api_python/python_gbsm.cpp` (created in R1 with the four conf functions; R4b adds the six), registered in
`python_main.cpp` as `py::module_ gbsm = m.def_submodule("gbsm", "Geometry-based stochastic model");`; the file is
added to the source list in `CMakeLists.txt` (line ~639, after `python_tools.cpp`). Mandatory arrays use
`.none(false)` as in `python_sos.cpp`.

```
phase = quadriga_lib.gbsm.lsf_init_phases( scenpar, seed=-1, rx_pos=None, tx_pos=None, dual_mobility=False,
    remove_bias=True, preset="Comb256", horizontal=True )
rnd = quadriga_lib.gbsm.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id=None, dual_mobility=False,
    preset="Comb256", horizontal=True )
pl, sf_sigma = quadriga_lib.gbsm.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d3d_in=None, rnd_he=None )
# complex=True (default)
lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( scenpar, fc, tx_pos,
    rx_pos, rnd, sf_sigma=None, vs_ratio=0.0, cholesky=True, clip_spreads=True, complex=True )
# complex=False
lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( ...,
    complex=False )
p_los, los_soft = quadriga_lib.gbsm.los_probability( model, d2d_out, h_bs, h_ut, d_clutter=nan, r_clutter=nan,
    h_clutter=nan, rnd_g=None, fc=None )
# [pending user decision R1-10]
loss, d2d_in, model_index, d3d_in, loss_el = quadriga_lib.gbsm.o2i_loss( fc, rnd, models, weights,
    d2d_in_max=25.0, tx_pos=None, rx_pos=None, d2d_in_min_of_two=True )
```

- `scenpar`, `plpar`: `py::handle` (dict or None). `seed`: `long long`. Flags: plain `bool` (SPEC_PYTHON_API
  scalars). `d_clutter`, `r_clutter`, `h_clutter`: `double` with default `std::numeric_limits<double>::quiet_NaN()`
  (Usage and docs say `nan`) [W-25].
- Outputs: `phase` `(n_coeff, 11, 2)` copied after the call; all others pre-allocated with
  `qd_python_init_output`; `model_index` uint64 (pre-allocated, C++ type kept, 0-based); `abs_toa` 1D `(n_link,)`;
  `gr_eps` complex128 `(n_link, n_freq)` built with `qd_python_copy2numpy<double, std::complex<double>>`; `los_soft`
  is `None` when `rnd_g` or `fc` is None; `d3d_in` and `loss_el` are `None` without positions [W-25].
- `models`: `qd_python_list2vector_Strings` (str or list of str). `floor_id`: float array, checked, converted [W-31].
- Each Python test covers the checklist of SPEC_PYTHON_API "What every test covers"; the specific points of 6.1
  apply with 0-based indices (`model_index` starts at 0) and these additions: keyword call with all Usage names;
  C-ordered, Fortran-ordered and strided `rx_pos` give the same result; `float32` positions accepted; `complex`
  flag of `lsp_generate` (tuple length 6 vs 7, `gr_eps.real == gr_eps_re`); `quadriga_lib.gbsm.conf_file_read` of a
  v19 file feeds `path_loss` and `lsp_generate` directly; `TypeError` for a missing mandatory argument and an
  unknown keyword; `ValueError` from C++ for an unknown path-loss model.


## 7. Implementation runs

Seven sequential runs, each one agent session [R1-06]: R1, R2a, R2b, R3a, R3b, R4a, R4b. A run starts only after the
previous run's report was accepted and Stephan committed. Every run reads the sections of this spec named in its
"Read" line (always including 1, 9 and the parts of 2.1 it adds), `md/SPEC_CPP_API.md` and, for wrappers, the
wrapper SPEC file of its API. The constraints of section 9 apply to every run. Each run adds to
`include/quadriga_gbsm.hpp` only the declarations it defines (2.1) [R1-27].

Every run writes `md/reports/r<N>_report.md` (N = 1, 2a, 2b, 3a, 3b, 4a, 4b) with the sections: **Summary** (what was
done, 5 lines); **Files** (created / changed); **Decisions and deviations** (every choice made within the spec,
every deviation from the spec or a SPEC file, with the spec section; the SPEC review points A/B/C are answered here
instead of stopping); **Test results** (commands and the actual summary lines of the runs: Catch2 test-case and
assertion counts, MEX suite result per MATLAB/Octave version, pytest summary, the objdump gate output where
required, load average before heavy runs); **Warnings** (compiler warnings, none expected); **Open points**
(proposals for the spec, things not tested); **Suggested commit cut** (themes and file lists; the agent does not
commit).

### R1 — header move, new keys, Python submodule

Read: 1, 2.1, 3.7, 4.1, 4.2, 4.4.1, 5.8 (R1 part), 6 (general rules), 9.

Files: `include/quadriga_gbsm.hpp` (new: guard, includes, the moved declarations and the comment block of the
"Large-scale fading" group, no new declarations), `include/quadriga_tools.hpp`, `include/quadriga_lib.hpp`,
`src/qd_scenpar.cpp`, `api_mex/conf_file_read.cpp`, `api_mex/conf_file_write.cpp`, `api_mex/conf_file_list.cpp`,
`api_mex/scenpar_validate.cpp`, `api_python/python_gbsm.cpp` (new), `api_python/python_tools.cpp`,
`api_python/python_main.cpp`, `api_python/qpy_conf_file_read.cpp`, `qpy_conf_file_write.cpp`,
`qpy_conf_file_list.cpp`, `qpy_scenpar_validate.cpp`, `CMakeLists.txt`, the existing tests of 5.8 (Catch2, MEX,
Python), 18 v19 files (4.4.1), `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`.

Steps:
1. Create the header, remove the block from `quadriga_tools.hpp`, include the new header in `quadriga_lib.hpp`;
   `src/qd_scenpar.cpp` and the conf tests include `quadriga_gbsm.hpp`. Build (Catch2 enabled).
2. Section "Geometry-based stochastic model" in the SECTION statements of `src/qd_scenpar.cpp`, the four MEX and
   the four Python wrapper files.
3. Key table (4.1, with its comment line), cross-key check (4.2), the DS overview line of the writer (4.1),
   documentation counts and lists (165 -> 171, 27 -> 33, 166 -> 172, the new keys in the `scenpar_validate`
   value-check list, the V/S term in the `conf_file_write` overview item).
4. Edit the v19 files (4.4.1).
5. Update the Catch2 tests (5.8, R1 part), run the conf tests, then the full suite.
6. Python: `python_gbsm.cpp` with the four includes and registrations moved from `python_tools.cpp` (identical
   `m.def` blocks), `python_main.cpp`, `CMakeLists.txt`; Usage lines `quadriga_lib.gbsm.<name>`; Python tests.
7. MEX tests (counts, cross-key error); MEX documentation counts.
8. SPEC files: `SPEC_CPP_API.md` lists `quadriga_gbsm.hpp` under `include/` and the section name; `SPEC_MATLAB_API.md`
   and `SPEC_PYTHON_API.md` add the section name; `SPEC_PYTHON_API.md` adds the submodule row
   `Geometry-based stochastic model | gbsm`, `python_gbsm.cpp` to the dispatcher list (7 dispatcher files) and
   `gbsm` to the submodule list.
9. Full gates (9.2).

Acceptance: `make test` passes (all Catch2 cases), no warnings; MEX suite passes in Octave and MATLAB R2024a; objdump
gate shows nothing above `GLIBCXX_3.4.28`; `make python` and `make python_test` pass; `quadriga_lib.tools` has no
conf functions, `quadriga_lib.gbsm` has the four.

### R2a — internal header, `lsf_init_phases`, `lsf_draw`

Read: 1, 2 (2.1 declarations of the two functions, 2.2), 3.0, 3.1, 3.2, 5.0 to 5.2, 5.7, 8, 9; source files
`include/quadriga_sos.hpp`, `src/sos_val.cpp` (device modes, kernel selection), `src/sos_remove_bias.cpp`,
`src/sos_preset.cpp` (interface only), the SOS helpers in `src/quadriga_lib_helper_functions.hpp`.

Files: `src/quadriga_gbsm_helpers.hpp` (new: documentation block, `qd_gbsm_gen`, `qd_gbsm_lsf_generators`,
`qd_gbsm_freq`, `qd_gbsm_positions`), `src/lsf_init_phases.cpp`, `src/lsf_draw.cpp` (new),
`include/quadriga_gbsm.hpp` (two declarations), `tests/catch2_tests/test_lsf_init_phases.cpp`, `test_lsf_draw.cpp`
(new), `tests/catch2_tests/qd_test_diag.hpp` (area line `QD_DIAG_GBSM`).

Steps (SPEC_CPP_API phases, without stops): declarations and stubs (section statement, first documentation block,
outputs sized and zeroed), build; the helper header; the tests of 5.1 and 5.2; run against the stubs and record in
the report which checks fail for the right reason; implement `lsf_init_phases`, then `lsf_draw` (file-local
evaluation, coupling and floor offsets written generically over the generator table, 3.0.5); iterate until green;
final documentation blocks (3.1, 3.2); full suite.

Acceptance: all new cases pass; `make test` passes; no warnings; documentation blocks complete. No MEX or Python
build needed.

### R2b — `lsp_generate`, shared LSP mean/std and Cholesky

Read: 1, 2 (2.1 declaration), 3.0, 3.4, 5.0, 5.4, 5.7, 8, 9; `src/quadriga_gbsm_helpers.hpp` and the
documentation blocks written in R2a; `qd_scenpar_xcorr` in `src/qd_scenpar.cpp`.

Files: `src/quadriga_gbsm_helpers.hpp` (add `qd_gbsm_lsp_mu_sigma`, `qd_gbsm_chol8`), `src/lsp_generate.cpp` (new;
the legacy root file-local), `src/qd_scenpar.cpp` (`qd_scenpar_xcorr` calls `qd_gbsm_chol8`),
`include/quadriga_gbsm.hpp` (one declaration), `tests/catch2_tests/test_lsp_generate.cpp` (new).

Steps: declaration and stub, build; the helpers; the tests of 5.4; run against the stub; implement; refactor
`qd_scenpar_xcorr` and run the conf tests; documentation block (3.4.4); full suite.

Acceptance: all new cases pass; the existing conf tests pass unchanged; `make test` passes; no warnings.

### R3a — `path_loss`, path-loss keys and constants

Read: 1, 2 (2.1 declaration), 3.0, 3.3, 4.3, 4.4.2, 5.0, 5.3, 5.8 (R3a part), 9, 11.1; `qd_conf_pl_comments` in
`src/qd_scenpar.cpp`; `src/quadriga_gbsm_helpers.hpp`.

Files: `src/path_loss.cpp` (new), `include/quadriga_gbsm.hpp` (one declaration), `src/qd_scenpar.cpp`
(`qd_conf_pl_comments` rows), the v19 files of 4.4.2, `tests/catch2_tests/test_path_loss.cpp` (new),
`test_conf_files_v19.cpp`, `test_conf_file_write.cpp` (comment rows).

Steps: declaration and stub, build; the tests of 5.3; run against the stub; the file edits of 4.4.2 before the TR
test is run against the implementation (it needs the full-precision constants, `PL_BP2D` and the other new keys);
implement; the writer comment rows and their test; the `test_conf_files_v19.cpp` updates; documentation block
(3.3.7); full suite.

Acceptance: all new cases pass; `make test` passes; no warnings.

### R3b — `los_probability`, `o2i_loss`

Read: 1, 2 (2.1 declarations, 2.3), 3.0, 3.5, 3.6, 5.0, 5.5, 5.6, 9, 11.2, 11.3.

Files: `src/los_probability.cpp` (new; the probit file-local), `src/o2i_loss.cpp` (new), `include/quadriga_gbsm.hpp`
(two declarations), `tests/catch2_tests/test_los_probability.cpp`, `test_o2i_loss.cpp` (new).

Steps: declarations and stubs, build; the tests of 5.5 and 5.6; run against the stubs; implement; documentation
blocks (3.5.5, 3.6.3); full suite.

Acceptance: all new cases pass; `make test` passes; no warnings.

### R4a — MEX wrappers, documentation sweep

Read: 1, 2, 3 (documentation-block items), 6 (general rules, 6.1), 7, 9, `md/SPEC_MATLAB_API.md`, and the C++
documentation blocks written in R2a to R3b.

Files: `api_mex/lsf_init_phases.cpp`, `lsf_draw.cpp`, `path_loss.cpp`, `lsp_generate.cpp`, `los_probability.cpp`,
`o2i_loss.cpp` (new); `tests/mex_tests/test_<name>.m` (6 new); `SECTION_DESC` blocks in `src/qd_scenpar.cpp` and
`api_mex/conf_file_read.cpp` [W-26]; `See also` lines of the ten C++ blocks (each new function names its
neighbours in the data flow of 1.1).

The `SECTION_DESC` (C++ version; MEX and Python adapt the names) explains in 10 to 20 lines: what a geometry-based
stochastic model is (TR 38.901 clause 7.5 steps 1 to 4 here), the conf files (`data/config`, 23 v19 files), the
data flow of 1.1 with the function names as `[[links]]` (incl. the layout order `o2i_loss` before
`los_probability`), the generator index of 2.2, the units [D5], and that the LOS state and O2I draws are made by the
caller with [[sos_rand]] / [[sos_randn]] at the distances of the keys `LOS_state_lambda`, `IO_state_lambda`.

Steps: MEX wrappers and tests (SPEC_MATLAB_API order: wrapper, then test); `SECTION_DESC` (C++ and MEX); `See
also` lines; full gates.

Acceptance: MEX suite passes in Octave and MATLAB R2024a (R2026a if available), objdump gate clean; `make test`
passes; no warnings in any build.

### R4b — Python wrappers, release notes

Read: 1, 2, 6 (general rules, 6.2), 7, 9, `md/SPEC_PYTHON_API.md`, the C++ and MEX documentation blocks.

Files: `api_python/qpy_<name>.cpp` (6 new), `api_python/python_gbsm.cpp` (registrations),
`tests/python_tests/test_<name>.py` (6 new), `SECTION_DESC` in `api_python/qpy_conf_file_read.cpp` [W-26].

Steps: Python wrappers, registrations, tests; `SECTION_DESC` (Python); release-note text (new functions, header
move, Python submodule move `tools` -> `gbsm` (breaking), new keys, changed v19 files incl. the path-loss changes of
up to 0.034 dB, the bias-removal change against QuaDRiGa) in the report [W-38]; full gates.

Acceptance: `make python` and `make python_test` pass; `make test` passes; no warnings in any build.


## 8. Requirements from the later SSF port

Read: `gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `get_channels.m` (lines 380 to 480), `add_sdc.m` and TR 7.5 steps 5
to 12, 7.6.2, 7.6.3.1, 7.6.13 to 7.6.16. The small-scale step needs from the LSF module:

| Quantity needed by the SSF step | Where (MATLAB) | Provided by | Status |
|---|---|---|---|
| Per-link, per-frequency DS, ASD, ASA, ESD, ESA (linear, deg) | `gen_ssf_parameters.m` lines 243 to 248 (`h_builder.ds` ...) | `lsp_generate` `lsp` slices 0, 3 to 6 | provided |
| KF per link and frequency (linear; TR steps 5 to 7 use K in dB) | line 248; TR (7.5-3), (7.5-8), (7.5-15) | `lsp` slice 1; dB = `10·lg` | provided |
| SF per link and frequency (step 12) | `get_channels.m` 428 to 480 | `lsp` slice 2 with `sf_sigma` from `path_loss` [D16] | provided |
| XPR mean per link and its std per frequency (step 9; QuaDRiGa uses the per-link XPR as mean) | lines 1048 to 1066 (`lsp_vals(8,2)`, `10·log10(xpr)`) | `lsp` slice 7 (mean in dB = `10·lg`), `lsp_sigma` slice 7 (std) | provided |
| LSP means without randomness (`lsp_vals`) | `add_sdc.m`, `gen_ssf_parameters.m` | `lsp_mu`, `lsp_sigma` (call with `rnd = zeros`) | provided (3.4.4), not 1:1: `lsp_vals` is the per-builder coefficient table without distance and height terms (e.g. `add_sdc` uses `mean(lsp_vals(5,1,:))`), while `lsp_mu` is per link with all terms; NG evaluates `lsp_generate` at a reference geometry where it needs a per-builder value [R1-22] |
| Per-link ZSD mean for the cluster ESD `c_ZSD = 3/8 · 10^mu_lgZSD` (7.6-7, key `PerClusterES_D_scale`) | not yet in NG | `lsp_mu` slice 5 (after the ESD terms and the floor) | provided |
| EoD offset (step 7, eq. 7.5-19 `+ mu_offset,ZOD`) | not in NG today | `eod_offset_deg` (elevation, positive up) | provided |
| Absolute TOA (7.6.9, NLOS only) | lines 813 to 818 | `abs_toa` | provided |
| Ground permittivity per link and frequency (7.6.8 extension) | lines 112, 262 to 265 | `gr_eps_re`, `gr_eps_im` | provided |
| Soft LOS state (7.6-19 combination) | — | `los_probability` `los_soft` | provided; the combination of H_LOS and H_NLOS is SSF and needs per link a LOS and an NLOS builder (two scenpar sets) with independent LSF phase sets, i.e. two `lsf_init_phases` calls with different seeds [R1-22] |
| Cluster-number range D1, D2 (7.6.15) | — | keys `NumClusters_min`, `NumClusters` | provided; the per-link draw is an SSF generator |
| Per-frequency handling (7.6.5: same x, independent per-cluster shadowing) | `xpr_sos`, `pin_sos` per frequency | LSF: one `x` for all frequencies; SSF: own generators | provided for LSF |
| Anchor-frequency LSPs of the alternative multi-frequency method (7.6.5.1: DS0, ASD0, ASA0, ZSD0, ZSA0 at e.g. 2 GHz) | — | `lsp_generate` with the anchor frequency added to `fc` (same `x`, so the anchor and the frequencies of interest are consistent); `clip_spreads` applies to the anchor as well, since 7.6.5.1 takes the spreads "according to Step 4", which includes the clipping | provided [R1-22] |
| Path loss for step 12 | `get_pl` | `path_loss` | provided |
| Spatially consistent cluster and ray variables (7.6.3.1: delays, shadowing, offsets, signs, coupling, XPR, phases; SC_lambda; same floor rule; mirrored for dual mobility) | `init_sos.m` lines 168 to 237 (`path_sos`, `xpr_sos`, `pin_sos`, `clst_dl_sos`) | **not** the LSF phase set (fixed 11 generators, `[n_coeff, 11, 2]`); the SSF port defines its own phase sets and evaluates them with the evaluation and bias removal of `lsf_draw.cpp` / `lsf_init_phases.cpp`, which R2a writes generically over the generator table (`qd_gbsm_gen`: lambda, distribution, mirror partner, bias flag) and which the SSF port moves into `src/quadriga_gbsm_helpers.hpp` (3.0.5) | requirement on R2a [W-39] |
| Floor separation of cluster variables (7.6.3.3 last paragraph) | — | the floor offsets of 3.2.4 inside the generic evaluation | requirement on R2a |
| Dual-mobility flag and the Tx/Rx-side conventions | `h_builder.dual_mobility` | `dual_mobility` argument, slice convention of 2.2 | provided |
| O2I flag (step 7: mean ZOA 90 deg for O2I), LOS/NLOS state | scenario of the builder | builder level, not LSF | n/a |
| d3D for the propagation delay `d3D/c` (7.6-43) | geometry | caller | n/a |
| Per-UT `Pr_sns` (7.6.14), near-field and polarization-variability draws | — | SSF generators | n/a |

Answer to "must the LSF phase set be extensible for SSF generators": no. It stays fixed at 11 generators (contract
of 2.1); the SSF port gets its own `ssf_init_phases`-type function. What must be extensible is the internal
evaluation, which is why R2a writes the evaluation and the bias removal generic over the generator table (3.0.5).
No public API was added for SSF readiness; the only API additions of this spec are `loss_el` of `o2i_loss` and the
model names of [W-19] [pending user decision R1-10].


## 9. Constraints (into every run)

### 9.1 Code

- `double` only [D12]; C++17; no warnings with `-Wall -Wextra -Wpedantic`; must compile with MSVC (no GCC
  extensions, no VLAs); OpenMP 2.0 only (signed loop variables, `OMP_SAFE_CALL`), results independent of the number
  of threads. OpenMP is optional in the new code (only loops over links in `path_loss`, `lsp_generate`, `o2i_loss`
  qualify).
- No BLAS/LAPACK: no `arma::chol`, `eig_sym`, `sqrtmat`, `norm`, `dot`, `solve`, `inv`, `det`, `svd`; the 8x8
  Cholesky (`qd_gbsm_chol8`, shared with `qd_scenpar_xcorr`) and the legacy root are written by hand (3.4.3).
  Armadillo element access and products are fine.
- Precision [W-44]: functions are double; `lsf_draw` calls `sos_val<double>` with `use_kernel = 0` (AVX2 kernel
  with single-precision sin/cos where available); tests of `lsf_draw` values use the tolerances of 5.0.
- Lean code: reuse existing helpers (`qd_sos_rng`, `qd_sos_uniform`, `qd_sos_acf_interp`, `qd_sos_preset_lookup`,
  `scenpar_validate`); a helper used once is a lambda, a helper used several times within one file is
  `static inline` or in an anonymous namespace, `src/quadriga_gbsm_helpers.hpp` only holds helpers used across
  files (3.0.5).
- No internal references in produced files [R1-03]: no IDs of decisions, writer choices, issues or findings (D-,
  W-, I-, R1-), no section numbers of this spec or the plan, no `md/` paths, no local paths (e.g. of the TR
  references folder) and no QuaDRiGa-NG line numbers in any source, header, test, wrapper, conf file or
  documentation block. Cite TR 38.901 (clause, table, equation, note) or other published material, or state the
  rule itself.
- No `std::to_chars` for floating point (MATLAB R2024a ships libstdc++ up to `GLIBCXX_3.4.28`). Every run that
  builds MEX files runs the gate `objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u`
  (nothing above 3.4.28).
- No global mutable state; random numbers only from the explicit seed (`lsf_init_phases`) or the caller's values.
- quadriga-lib must not depend on QuaDRiGa-NG: no paths to it, no data derived from its files in sources or tests
  (formulas and tables typed from this spec, the MATLAB sources and the TR are fine). QuaDRiGa-NG and the TR are
  read-only.
- Agents do not commit, push or merge (D31). They do not edit this spec, the plan or the findings file; proposals go
  into the run report.

### 9.2 Resources and commands

- Check `cat /proc/loadavg` before heavy work; at most 12 threads (`CMAKE_BUILD_PARALLEL_LEVEL=12` for `make`
  targets, `cmake --build build_linux --parallel 12`, `OMP_NUM_THREADS=12` or less); never two builds or two test
  suites at the same time; heavy tools (ASan, MATLAB, sweeps) once on the final code of a run.
- Catch2: `make test` (configure with tests, build, run all); iterate with
  `cmake --build build_linux --parallel 12 && build_linux/test_bin "lsf_draw*"`; diagnostics with
  `QD_DIAG_GBSM=1 build_linux/test_bin "path_loss*"`.
- MEX: `make matlab=ON` / `make octave=ON`; single tests from MATLAB/Octave as in SPEC_MATLAB_API; MATLAB R2024a,
  R2026a and Octave can be run from the shell (e.g. `matlab -batch "..."`, `octave --eval "..."`; the run report
  states which binaries were used). The MEX suite runs in Octave and R2024a in R1 and R4a.
- Python: `make python`, `make python_test`, single files with `python3 -m pytest tests/python_tests/test_<name>.py -x -s`.
- No CUDA involved.


## 10. Integration hand-over (QuaDRiGa-NG, after the merge; out of scope for R1 to R4b)

Carried over from plan section 10, with the additions of this spec. Starts after the merge into `main`; NG becomes
writable then [D32]. The manager splits it into prompts:

- **15a Library update in NG:** copy the new quadriga-lib (as in the earlier ports: `external/`, the MEX
  `+quadriga_lib`), copy the 23 v19 conf files to `quadriga_src/config`.
- **15b Builder LSF:** the builder stores `lsf_phase` (`[n_coeff, 11, 2]`) instead of the 8 + 2 `qd_sos` objects in
  `sos`, `absTOA_sos`, `gr_sos` (`init_sos` LSF part replaced by `lsf_init_phases`; `copy`, `split_rx`,
  `split_multi_freq`, `add_paths` adapted); `gen_lsf_parameters`, `get_sf_profile`, `get_lsp_map` become thin
  wrappers of `lsf_draw` + `path_loss` + `lsp_generate`; `lsp_vals` replaced by the `lsp_mu` / `lsp_sigma` outputs
  where it is used (`add_sdc`, `gen_ssf_parameters`; call with `rnd = zeros` where no draw is wanted); `get_pl`
  becomes a wrapper of `path_loss` (the O2I loss of the track is added by the wrapper once, not twice as in today's
  `nlos` branch [I-7]); `get_channels` uses `sf_sigma` instead of `scale_sf` [D16]. `simpar` gets the legacy
  switches of D4 with a "legacy" preset:

  | Switch | 3GPP default | Legacy (today's QuaDRiGa) |
  |---|---|---|
  | `lsp_generate` `cholesky` [pending user decision R1-09] | true | false (sqrtm) |
  | `lsp_generate` `clip_spreads` | true | false |
  | `lsf_init_phases` / `lsf_draw` `horizontal` | true | false (3D) |
  | `path_loss` `d3d_in` | omitted (geometric distance) | given (outdoor distance) |
  | `o2i_loss` `d2d_in_min_of_two` [pending user decision R1-10] | true | false |
  | `o2i_loss` high-loss model [pending user decision R1-10] | `high` | `high_v14` |

  Not switchable, changes against today that NG documents: the 3GPP_3D groups (TR 36.873 UMi/UMa) use
  `38.901_UMi` / `38.901_UMa` (the `d^2` of `set_scenario.m` was a bug; indoor UTs above 13 m change); `winner_los`
  uses `C2` beyond the breakpoint and per-link breakpoint heights (the 5 WINNER LOS files change beyond the
  breakpoint); the bias removal acts after the mirroring, so bias-removed ASD/ASA and ESD/ESA change in dual
  mobility and, for different decorrelation distances of a pair, also in single mobility (3.1); the v19 path-loss
  constants and the RMa/SMa breakpoint form (up to 0.034 dB).

- **15c Layout:** `set_scenario` keeps a table group -> (LOS model name, member files, parameters) and calls
  `los_probability`; LOS and indoor state distances from `LOS_state_lambda`, `IO_state_lambda`; groups for the v19
  set incl. SMa and InF-HH; `gen_o2i_loss` replaced by `o2i_loss` (models and weights from the group,
  `d2d_in_max = O2I_d2D_in_max`, the mmMAGIC elevation term from `loss_el` [pending user decision R1-10: or
  mmMAGIC kept in MATLAB]) and deleted; `o2i_loss` before `los_probability`, `d2d_out = max(d2D - d2d_in, 0)`;
  the 3GPP_3D groups mapped to `38.901_UMi` / `38.901_UMa`; an in-car option for UTs
  (`car`, `car_metal`). The renormalisation of the LOS-state draw stays in NG as a legacy option [D24].
- **15d Parity test (D23), in two parts [R1-18]:** a MATLAB script computes, with the pre-integration code, the SOS
  values `val`, PL, SF and the 8 LSPs of the legacy files for single mobility at fixed phases and stores them in
  NG's test data.
  (1) `lsp_generate` and `path_loss` with legacy flags, fed with MATLAB's own SOS values as `rnd` (columns 0..7
  from `val`), reproduce PL, SF and the LSPs to 1e-12 relative. The 5 WINNER LOS files (`winner_los`) are compared
  only for links before the breakpoint, since the library uses `C2` beyond it (W-42); the breakpoint itself uses
  per-link heights, equal to MATLAB when the heights within one MATLAB call are constant.
  (2) `lsf_draw` against MATLAB's `val`: 1e-5 absolute, because `lsf_draw` evaluates with the AVX2 kernel
  (single-precision sin/cos) and the frequencies of `qd_sos` (`freq10 · maxD_old / maxD_new`) are not guaranteed
  to be bitwise equal to those of `sos_preset(..., lambda)`.
  The phase cube for the library is built from the old generators: `phase(:, g, 1) = h_builder.sos(g).sos_phase(:,
  1)` for g = 1..8 (MATLAB indices), `absTOA_sos` for column 9 (any phases if absent, the value is then unused),
  `gr_sos` for column 10, any phases for column 11 (hE; unused, legacy files have no `PL_hE_rand`); slice 2 by
  mirroring. The parity runs without bias removal in the library (the phases come from MATLAB, already corrected
  by MATLAB's rule), so the bias-removal difference of 3.1 does not enter.
- **15e NG unit tests and tutorials:** all `xunit_new` tests pass in MATLAB R2024a and Octave; the calibration
  tutorials `calib_3GPP_38901_large_scale.m` and `calib_3GPP_38901_full.m` run with the v19 groups.
- Retired MATLAB code (target): `gen_o2i_loss.m`; the LSF part of `init_sos.m`; most of `gen_lsf_parameters.m`,
  `get_pl.m`, `get_sf_profile.m`, `get_lsp_map.m`; the p_LOS formulas in `set_scenario.m`; the `lsp_vals` property.


## 11. Appendix: tables

### 11.1 Zenith atmospheric attenuation of the `satellite` model (`get_pl.m` lines 447 to 460)

93 frequencies in GHz:

```
1, 1.99, 2.97, 3.96, 4.95, 5.92, 6.87, 7.87, 8.83, 9.77, 11.03, 12.32, 13.17, 14.44, 15.65, 16.37, 17.18, 18.03,
18.8, 19.34, 19.91, 20.42, 20.71, 21.16, 21.49, 21.7, 22.09, 22.52, 22.99, 23.98, 24.93, 26.13, 27.38, 29.18,
30.44, 32.18, 33.8, 36.09, 38.87, 40.59, 42.33, 44, 44.96, 47.2, 48.42, 49.42, 50.17, 50.96, 51.73, 52.62, 53.32,
53.6, 53.75, 54.06, 54.38, 55.12, 55.73, 56.54, 56.97, 57.52, 58.01, 58.18, 58.33, 58.62, 58.99, 59.38, 59.57,
59.84, 60.27, 60.59, 60.93, 61.22, 61.44, 61.93, 62.27, 63.34, 64.27, 65.42, 66.35, 67.19, 68.16, 69.22, 70.33,
71.66, 73.53, 75.28, 77.48, 80.64, 83.77, 87.07, 92.1, 96.39, 100
```

93 zenith attenuations in dB:

```
0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09, 0.11, 0.13, 0.16, 0.19,
0.23, 0.28, 0.35, 0.4, 0.46, 0.51, 0.52, 0.52, 0.49, 0.44, 0.36, 0.3, 0.26, 0.24, 0.23, 0.24, 0.25, 0.27, 0.32,
0.4, 0.46, 0.56, 0.68, 0.8, 1.16, 1.53, 2.18, 2.99, 4.47, 8.08, 15.49, 30.37, 39.94, 47.33, 60.66, 80.97, 102.6,
120.55, 137.92, 144.01, 149.57, 153.68, 155.18, 156.63, 157.9, 158.81, 157.57, 156.04, 151.64, 141.33, 130.52,
119.21, 97.42, 68, 52.74, 39.81, 17.77, 8.61, 4.2, 2.8, 2.19, 1.69, 1.42, 1.22, 1.07, 0.93, 0.85, 0.79, 0.75,
0.74, 0.75, 0.81, 0.87, 0.97
```

### 11.2 LOS probability of the 38.811 models in percent (`set_scenario.m`)

Elevation grid in degrees: `-90, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90`; linear interpolation, constant beyond the
ends (elevations are always within [-90, 90]).

| Model | Values |
|---|---|
| `38.811_DenseUrban` | 0, 0, 28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1 |
| `38.811_Urban` | 0, 0, 24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2 |
| `38.811_Suburban`, `38.811_Rural` | 0, 0, 78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8 |

### 11.3 Legacy d2D-in distribution (`gen_o2i_loss.m`) [pending user decision R1-10: deleted if the O2I scope is reduced]

Distance fractions `0, 0.1, ..., 1` (11 values) at the cumulative probabilities
`0, 0.193, 0.36, 0.514, 0.642, 0.749, 0.84, 0.911, 0.962, 0.991, 1`. `Finv(u)` interpolates the fractions linearly
over the probabilities (`quadriga_lib.interp(cdf, [], val, u)` in MATLAB; `qd_sos_acf_interp` in the library);
`d2din = Finv(u) · d2d_in_max`.


## 12. Change log

Internal; IDs refer to `md/gbsm_lsf_findings.md`.

| Version | Change |
|---|---|
| r0 | First draft (spec writer). |
| r1 | Review round 1, findings R1-01 to R1-30. Precision rule and kernel tolerances for `lsf_draw` values (R1-01; 3.0.1, 3.2, 5.0, 5.1, 5.2, 9.1). RMa/SMa LOS follow the TR's 2D breakpoint form via the new key `PL_BP2D` instead of a 1e-2 dB oracle (R1-02; W-41, 3.3.4, 4.3, 4.4.2, 5.3). No internal references in produced files; header comments without decision IDs; no local paths in tests (R1-03; header of this spec, 2.1, 5.0, 9.1). Writer DS overview line with the V/S term, conf-key tests completed (R1-04; 4.1, 4.3, 5.8). Bias removal documented to act on one floor (R1-05; W-43, 3.1). Runs split into R1, R2a, R2b, R3a, R3b, R4a, R4b (R1-06; 7). `winner_los` with `C2` (R1-07; W-42). Internal header limited to cross-file helpers, shared Cholesky with `qd_scenpar_xcorr`, no internal test file (R1-08; 3.0.5, 5.7). Union-of-positions bias removal and per-field evaluation with exact swap reciprocity (R1-11; W-2, W-5, 3.1, 3.2). 36.873 LOS models and `indoor_ut` dropped (R1-12). `o2i_total_distance` dropped (R1-13; W-46). `scenpar_validate(..., true)` instead of per-function key checks (R1-14; W-29, 3.0.3). Per-link `winner_los` breakpoint (R1-15). Floor-offset property corrected, `qd_sos_rng` reused (R1-16; 2.3, 3.2.4). Bias quirk scope stated (R1-17; 3.1, 10). Parity test split (R1-18; 10). Lean `test_lsp_generate` (R1-19; 5.4). O2I before LOS probability, InF LOS-state distance note (R1-20; 1.1, 3.5.5). O2I SF oracle (R1-21; 5.3). SSF table gaps (R1-22; 8). Helper reuse `qd_sos_acf_interp` (R1-23). Probit file-local, relaxed target, tested through `los_soft` (R1-24; 3.5.4, 5.5). `sf_sigma >= 0` (R1-25). Key-table comment (R1-26). Declarations added per run (R1-27; 2.1, 7). Inconsistencies (R1-28; 3.1, 3.2, 5.1, W-45). Permittivity sign note (R1-29; 3.4.2, 3.4.4). One value per `plpar` entry, D27 comment (R1-30; 3.3.1, 4.4.2). Pending user decisions marked: R1-09 (legacy root; 2.1, 3.4.2, 3.4.3, 5.4, 6, 10), R1-10 (O2I scope; 2.1, 2.3, 3.3.2, 3.6.0, 5.6, 6, 8, 10, 11.3). |
