# Plan: port the QuaDRiGa large-scale model to quadriga-lib

Status: **Phases 1 to 3 done; Phase 4 (manager hand-off) ready.** Written 2026-10-23 by the planning session.
Sections 1 to 7 are settled (API, decisions); section 8 is the implementation plan; section 9 the procedure of the
manager; section 10 the hand-over for the QuaDRiGa-NG integration; section 11 the status table that the manager
keeps current. No source file has been changed yet.

Branch `fixes` of `/sjc/quadriga-lib-fixes`, starting at cebd0c0 (v0.12.3). QuaDRiGa-NG (`/sjc/QuaDRiGa-NG`) and
the 3GPP reference (`/sjc/references/3gpp_38901`) are read-only until the integration phase (WP-15).

Read first, in this order: this document; `md/SPEC_CPP_API.md` (workflow, layout, tests, documentation blocks);
for wrappers `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md`.


## 1. Goal and phases

Port the large-scale part of QuaDRiGa to the C++ core:

- path loss and shadow-fading standard deviation (`qd_builder.get_pl`),
- the large-scale parameters DS, KF, SF, ASD, ASA, ESD, ESA, XPR (`qd_builder.lsp_vals`, `init_sos` LSF part,
  `gen_lsf_parameters`, `get_sf_profile`, `get_lsp_map`),
- the quantities drawn together with them: absolute TOA offset, ground-reflection permittivity, EoD offset,
- LOS probability and soft LOS state (`qd_layout.set_scenario`, probability part),
- O2I penetration loss (`qd_layout.gen_o2i_loss`).

Goal 1: full coverage of the large-scale model of 3GPP TR 38.901 v19.5.0 (clauses 7.4, 7.5 steps 2 to 4,
7.6.3.3, 7.6.9 and the LSF-related parts of 7.6.5, 7.6.10).
Goal 2: replace the MATLAB implementations in QuaDRiGa-NG by library calls and retire as much MATLAB code as
possible.

| Phase | Content | Owner | Work packages |
|---|---|---|---|
| 1 | Read sources and TR | planning session | done |
| 2 | Public C++ API, designed with Stephan | planning session | done (sections 3 to 7) |
| 3 | Implementation plan | planning session | done (section 8) |
| 4 | Manager set-up and hand-off | planning session | section 9 |
| 5 | Implementation (test-first, SPEC_CPP_API) | workers, run by the manager | WP-0 to WP-7 |
| 6 | MEX and Python wrappers with tests | workers | WP-8, WP-9 |
| 7 | Two review rounds plus fixes | review workers, manager | WP-10 to WP-13 |
| 8 | Merge into `main` | manager, on request | WP-14 |
| 9 | Integration into QuaDRiGa-NG | integration workers, manager | WP-15 (section 10) |

The parser (`conf_file_read`, `scenpar_validate`, `conf_file_list`, `conf_file_write`) and the SOS functions
(`sos_*`) are already ported and are used as they are.


## 2. Sources read (Phase 1)

- quadriga-lib: `include/quadriga_tools.hpp` (scenario parameters), `include/quadriga_sos.hpp`,
  `src/qd_scenpar.cpp` (165 keys, 27 of them for v19.5.0 and unused so far), `data/config/3GPP_38.901_v19_*.conf`
  (23 files), the plan of the parser port (`git show 04acbc0^:md/conf_file_parser.md`; section 5 is the first
  38.901 v19.5.0 gap analysis, section 7.2 the formulas of the v19 keys).
- QuaDRiGa-NG `quadriga_src/@qd_builder`: `qd_builder.m` (`lsp_vals`, `lsp_xcorr`), `gen_parameters.m`,
  `init_sos.m`, `gen_lsf_parameters.m`, `get_pl.m`, `get_sf_profile.m`, `get_lsp_map.m`, the use in
  `get_channels.m` (lines 428 to 480), `add_sdc.m`, `gen_ssf_parameters.m`, `check_los.m`, `gen_lsf_from_ssf.m`;
  `@qd_layout/set_scenario.m`, `gen_o2i_loss.m`; `@qd_sos/qd_sos.m`.
- TR 38.901 v19.5.0 (`38901-v19.5.0.md`): 7.2 (Tables 7.2-4, 7.2-5), 7.4.1 to 7.4.4, 7.5 steps 1 to 4,
  Tables 7.5-6 (Parts 1 to 4) and 7.5-7 to 7.5-12, 7.6.3.1, 7.6.3.3, 7.6.3.4, 7.6.5, 7.6.9, 7.6.15, 7.6.16,
  7.8.4 (InF calibration).


## 3. Decisions

Answers given by Stephan on 2026-10-23. Binding unless he changes them. The manager records later decisions at the
end of this table (D-numbers continue).

| # | Question | Answer |
|---|---|---|
| D1 | Scope besides PL, SF and the 8 LSPs | **All of:** LOS probability and soft LOS state; O2I penetration loss; absolute TOA and ground-reflection permittivity; EoD offset and the new ESD terms. |
| D2 | Path-loss representation | **Port the plpar models and extend them.** All 10 parametric models of `get_pl.m` 1:1; the missing 38.901 behaviours as new `PL_` keys. |
| D3 | Random state | **Phase arrays in and out.** Free functions; the builder stores plain arrays; re-evaluation along tracks and on maps works with the stored phases. |
| D4 | Policy where TR and QuaDRiGa differ | **Both supported, 3GPP behaviour is the default.** QuaDRiGa-NG passes legacy flags where it wants the old behaviour. |
| D5 | Units of the outputs | **As the builder:** DS in s, KF, SF and XPR as linear power ratios, AS and ES in degrees, PL in dB. Arguments in degrees carry `_deg` in their name or say so in the comment. Approved exception to SPEC_CPP_API. |
| D6 | Spatial correlation of LSPs | **2D (horizontal plane) plus an optional floor ID**; different floors are uncorrelated (Table 7.5-6 note 3). 3D (legacy) is an option. |
| D7 | LOS probability models | **Named models in C++.** Model parameters are function arguments with the TR values as defaults. The group table (group name to model and member files) stays in QuaDRiGa-NG `set_scenario`, reduced to names. |
| D7a | Conf format | **Fixed.** No new file type, no new string keys, no change of the reader and writer interfaces. Adding numeric keys to the key table and new `PL_` keys is allowed. |
| D8 | Group-level numeric settings | LOS-state and indoor-state decorrelation distances and the per-scenario O2I constants are **numeric scenpar keys** in the files where they are unique (NLOS files, O2I files). InF and SMa geometry (d_clutter, r, h_c, vegetation) are **arguments** of the named LOS models. |
| D9 | Input of the scenario | **`scenpar` and `plpar` maps**, exactly as `conf_file_read` returns them. |
| D10 | Header | **New public header `include/quadriga_gbsm.hpp`** ("geometry-based stochastic model"). `conf_file_read`, `scenpar_validate`, `conf_file_list`, `conf_file_write` move there from `quadriga_tools.hpp` (clean move; `quadriga_lib.hpp` includes the new header). Later SSF and other QuaDRiGa-NG functions go there too. |
| D11 | Documentation section | **"Geometry-based stochastic model"**, for the new functions and the four moved ones (they leave "Channel generation functions"). |
| D12 | Numeric type | **`double` only**, no template. |
| D13 | ACF of the LSP generators | **`Comb256` default** (QuaDRiGa); the preset is an argument, `Exp256` gives the literal TR ACF. |
| D14 | Bias removal of the LSP phases | **Optional, on by default** whenever drop positions are passed. |
| D15 | UMa random environment height hE | **Spatially consistent SOS** uniform variable, generator 10 of the LSF phase set, decorrelation distance `SF_lambda` (D19). |
| D16 | SF standard deviation | **`path_loss` returns `sf_sigma` per link and frequency; `lsp_generate` takes it** as optional input; the returned SF is final. Replaces `scale_sf` in `get_channels`. |
| D17 | Function set and names | **As in section 6**: `lsf_init_phases`, `lsf_draw`, `path_loss`, `lsp_generate`, `los_probability`, `o2i_loss`. |
| D18 | LSP output layout | **One cube `[n_link, n_freq, 8]`**, slices DS, KF, SF, ASD, ASA, ESD, ESA, XPR. |
| D19 | Decorrelation distance of hE | **`SF_lambda`**, no new key. |
| D20 | Dual mobility | **Explicit `bool dual_mobility`** argument. |
| D21 | Bias removal and reciprocity | **Mirror first, then remove the bias with both devices.** Fixes the quirk of 5.2; dual-mobility values change against today. |
| D22 | Absolute TOA | **Keep the rule** "active if `absTOA_mu > -30`"; `abs_toa` = 0 otherwise; bound 2L/c when `absTOA_Lmax > 0`. |
| D23 | Legacy parity | **Yes, in the integration phase**: with legacy flags and the same phases the library reproduces today's MATLAB values to 1e-12 relative, single mobility only (D21 changes dual mobility). The reference values come from a MATLAB script in NG; nothing in quadriga-lib depends on NG. |
| D24 | Renormalisation of the LOS-state draw | **Not in the library.** NG may keep it as a legacy option. |
| D25 | O2I scope | **All of:** d3D-in geometry, mmMAGIC model, car loss, < 6 GHz model. `gen_o2i_loss.m` can be deleted after the integration. |
| D26 | InF delay spread | **Keys plus override:** `DS_VS_a`, `DS_VS_b`, `DS_VS` in the InF files; argument `vs_ratio` of `lsp_generate` overrides `DS_VS`. |
| D27 | Optional TR NLOS path-loss formulas | **Documented only** (comments in the v19 NLOS files give the `logdist` parameters); no new files. |
| D28 | Review points A, B, C of the SPEC | **Manager decides** within this plan; escalates to Stephan only contract changes, deviations from a decision here, and open TR questions. Stephan sees a short report per work package. |
| D29 | Python | **New submodule `quadriga_lib.gbsm`**; the four conf functions move there from `tools` (breaking for Python users, noted in the release notes). MATLAB names stay flat and unchanged. |
| D30 | Workers | **The manager writes each worker prompt** (with "manager defaults" for open choices); Stephan starts the worker session; the manager reviews and updates this plan. |
| D31 | Commits | **Stephan commits**; the manager prepares the cut (themed commits), the message files and the `git add` commands. |
| D32 | Integration | **Same manager, after the merge**: it writes the integration prompts; NG becomes writable then. |

Decisions taken over from the parser port and still valid: elevation convention (QuaDRiGa names ESD, ESA, EoD;
`elevation = 90 deg - zenith`; the EoD offset has the opposite sign of the TR's ZOD offset); the formulas of the
27 v19 keys (old plan section 7.2, repeated in 5.1 below); the v19 conf files live in `data/config/` under
Apache-2.0.


## 4. Gap analysis: TR 38.901 v19.5.0 against QuaDRiGa

Disposition: **lib** = implemented by this port; **key** = new conf key; **file** = fix in the v19 conf files;
**NG** = stays in QuaDRiGa-NG; **out** = not in scope. The WP column names the work package.

### 4.1 Large-scale parameters (7.5 step 4)

| # | TR | QuaDRiGa today | Disposition | WP |
|---|---|---|---|---|
| L1 | Cholesky root of C in the order SF, K, DS, ASD, ASA, ZSD, ZSA | `sqrtm(R)` in the order DS, KF, SF, ASD, ASA, ESD, ESA, XPR | lib: Cholesky in TR order (XPR last) default; `sqrtm` option, by a hand-written Jacobi eigen-decomposition (8x8, no LAPACK) | 4 |
| L2 | Clip ASA, ASD to 104 deg, ZSA, ZSD to 52 deg | not clipped | lib: clip default, option off | 4 |
| L3 | fc = 6 GHz below 6 GHz (UMa; InH ZSD/ZOD), fc = 2 GHz below 2 GHz (UMi) | not modelled | lib: consume `LSP_fc_min` | 4 |
| L4 | ZSD mean depends on hUT, abs(hUT - hBS), max(hUT - hBS, 0) | only `ES_D_mu_A`, `ES_D_mu_min` | lib: consume `ES_D_mu_B/C/D` | 4 |
| L5 | ZOD offset (Tables 7.5-7 to 7.5-12) | not modelled | lib: `eod_offset_deg` output from `EoD_offset_*`; consumed by the later SSF port | 4 |
| L6 | Correlation in the horizontal plane; no correlation between floors | 3D correlation | lib: 2D + floor ID default, 3D option (D6) | 2 |
| L7 | LSPs of different BS sites uncorrelated, co-sited sectors equal | one generator set per builder (= site and scenario) | already met; documented | 2 |
| L8 | InF: lgDS mean = lg(26 V/S + 14) - 9.35 (LOS), lg(30 V/S + 32) - 9.44 (NLOS) | evaluated for one hall and written as a number | key + lib (D26) | 0, 4 |
| L9 | Multi-frequency: same x for all frequencies (7.6.5) | same | lib: kept | 4 |
| L10 | Dual mobility (7.6.10 defines Doppler only) | ASD/ASA and ESD/ESA coupled by the ACF at the Tx-Rx distance | lib: kept (QuaDRiGa extension) | 2 |

### 4.2 Path loss (7.4.1, 7.4.4)

| # | TR | QuaDRiGa today | Disposition | WP |
|---|---|---|---|---|
| P1 | UMa hE = 1 m w.p. 1/(1 + C(d2D, hUT)), else uniform from {12, 15, ..., hUT - 1.5} | fixed `PL_hE` | lib + `PL_hE_rand` (D15) | 3 |
| P2 | InF-DL: max(PL_DL, PL_LOS, PL_SL) | max of two terms | lib + `PL_Ax/Bx/Cx` | 3 |
| P3 | SMa NLOS: no max with LOS | the v19 file sets the LOS part to 0 | lib + `PL_LOSmax`; file cleaned up | 3 |
| P4 | RMa/SMa break point 2 pi hBS hUT fc / c | fixed in the v19 files (`PL_E = 20.944`, `PL_hE = 0`) | done (file) | - |
| P5 | O2I: PL_b uses d3D-out + d3D-in (total distance) | outdoor distance only (`get_pl` subtracts d3D-in) | lib: TR default, legacy option | 3 |
| P6 | SF std depends on the breakpoint (RMa, SMa LOS: 4 / 6 dB); O2I < 6 GHz model: 7 dB | `scale_sf` applied in `get_channels` | lib: `sf_sigma` output (D16) | 3 |
| P7 | Optional UMa/UMi/InH NLOS formulas | possible with `logdist` | documented in the v19 NLOS files (D27) | 3 |
| P8 | Autocorrelation of SF exponential (7.4-5) | Comb256 | D13 | - |

### 4.3 LOS probability and states (7.4.2, 7.6.3.3)

| # | TR | QuaDRiGa today | Disposition | WP |
|---|---|---|---|---|
| S1 | Table 7.4.2-1 incl. InF (SL, DL, SH, DH, HH) and SMa | switch in `set_scenario` (no SMa, no InF-HH) | lib: named models (D7) | 5 |
| S2 | Spatially consistent LOS state, correlation distance per scenario (Table 7.6.3.1-2) | hard-coded per group in `set_scenario` | key `LOS_state_lambda`; draw stays a `sos_rand` call in NG | 0 |
| S3 | Indoor state, correlation distance 50 m | `set_scenario` | key `IO_state_lambda` | 0 |
| S4 | Soft LOS (7.6-18) | not modelled | lib: `los_soft` output; combining H^LOS and H^NLOS (7.6-19) is SSF, out | 5 |
| S5 | InF calibration (7.8.4): r = 20 % / 60 %, h_c = 2 / 6 m, d_clutter = 10 / 2 m for low / high density | `set_scenario` uses r = 0.4 for InF-DH | lib: TR calibration values as defaults | 5 |

### 4.4 O2I penetration loss (7.4.3)

| # | TR | QuaDRiGa today | Disposition | WP |
|---|---|---|---|---|
| O1 | IRR glass 25.4 + 0.11 f | 23 + 0.3 f | lib | 6 |
| O2 | Low-loss A model (glass + plywood), SMa | missing | lib | 6 |
| O3 | d2D-in max 25 m (UMa, UMi, SMa commercial), 10 m (RMa, SMa residential) | argument of `gen_o2i_loss` | key `O2I_d2D_in_max` in the O2I files; argument | 0, 6 |
| O4 | Building type, d2D-in, sigma_P spatially consistent (50 m, 25 m, 10 m) | one SC distance for all | caller draws with `sos_rand`/`sos_randn` at the TR distances | NG |
| O5 | Car penetration loss N(9, 5), N(20, 5) for metallized windows | missing | lib | 6 |
| O6 | Backwards-compatible model < 6 GHz (Table 7.4.3-3) | missing | lib | 6 |
| O7 | mmMAGIC O2I model | `gen_o2i_loss` method 'mmMAGIC' | lib (D25) | 6 |
| O8 | d2D-in = min of two uniforms | inverse CDF of min(U1, U2) applied to one uniform | lib: min of two uniforms (TR) default; legacy option | 6 |

### 4.5 Absolute TOA and ground reflection

| # | TR | QuaDRiGa today | Disposition | WP |
|---|---|---|---|---|
| T1 | Delta tau log-normal, Table 7.6.9-1, NLOS only | `absTOA_*` keys | lib (in `lsp_generate`, D22) | 4 |
| T2 | Upper bound 2L/c for InF and InH | missing | lib: consume `absTOA_Lmax` | 4 |
| R1 | Ground permittivity (QuaDRiGa extension, model of 7.6.8) | drawn in `gen_lsf_parameters` | lib (in `lsp_generate`) | 4 |

### 4.6 Out of scope of this port

Cluster-number variability (7.6.15), near field (7.6.13), spatial non-stationarity (7.6.14), polarization
variability (7.6.16), soft-LOS channel combination (7.6-19): small-scale fading, later SSF port; the keys exist
already. Blockage (7.6.4), ISAC (7.9), map-based model (8): out.


## 5. How QuaDRiGa computes it today (reference for the port)

### 5.1 LSP mean and standard deviation (`qd_builder.lsp_vals`, `gen_lsf_parameters`)

For LSP i (DS, KF, SF, ASD, ASA, ESD, ESA, XPR) at frequency f [GHz], link with d2D, hBS (Tx z), alpha =
atan(hBS / d2D), d2D and hBS clamped to >= 1e-4:

```
mu_i    = mu + gamma · lg(omega + f) + epsilon · lg(d2D) + zeta · lg(hBS) + alpha_i · lg(alpha)
sigma_i = max(0, sigma + delta · lg(omega + f)) + kappa · lg(d2D) + tau · lg(hBS) + beta · lg(alpha)
ESD:      mu += ES_D_mu_A · d2D / 1000;  mu = max(mu, ES_D_mu_min)
value   = 10^(mu_i + sigma_i · x_i)             (DS in s, AS and ES in deg)
          10^(0.1 · (mu_i + sigma_i · x_i))     (KF, SF, XPR; mu and sigma in dB)
```

SF has no mu and no gamma, epsilon, zeta, alpha (zero). x = R^(1/2) · s, s the 8 SOS values N(0,1) at the link.

v19 additions (old parser plan, section 7.2): `f` is replaced by `max(f, LSP_fc_min)` in all frequency terms of
the LSPs and of the EoD offset; the ESD mean gets `+ ES_D_mu_B · hUT + ES_D_mu_C · abs(hUT - hBS) + ES_D_mu_D ·
max(hUT - hBS, 0)` before the floor `ES_D_mu_min`.

```
EoD offset [deg, positive = up] =
      EoD_offset_e + EoD_offset_e_gamma · lg(fc)
    + 10^( (EoD_offset_a + EoD_offset_a_gamma · lg(fc)) · lg(max(EoD_offset_b, d2D))
           + EoD_offset_c + EoD_offset_c_gamma · lg(fc) + EoD_offset_c_hUT · hUT )      if EoD_offset_b > 0
    + atan((hcl - hBS)/d2D) - atan((hUT - hBS)/d2D)  [in deg]                           if hcl > 0 and hUT < hcl
```

with `fc = max(f, LSP_fc_min)` in GHz and `hcl = EoD_offset_hcl`.

### 5.2 SOS generators (`init_sos`)

8 generators with the LSP decorrelation distances, preset from `simpar.autocorrelation_function`; two phase sets
(device A = Rx, device B = Tx). Reciprocity: DS, KF, SF, XPR use the same phases for both devices; ASD of device A
= ASA of device B and vice versa; ESD/ESA likewise. Bias removal over the Rx positions (and the Tx positions in
dual mobility) when there are more than 2. Absolute TOA: one more Normal generator (`absTOA_lambda`, only if
`absTOA_mu > -30`). Ground reflection: one Uniform generator with `SC_lambda`.
Quirk: the bias is removed before the phases are mirrored, so for dual mobility the mirrored device loses its
correction (fixed by D21).

### 5.3 Path loss (`get_pl`)

Models `logdist`, `logdist_simple`, `constant`, `winner_los`, `winner_nlos`, `winner_pathloss`, `dual_slope`,
`tripple_slope`, `nlos` (max of a dual-slope LOS term and a 13-term NLOS formula), `satellite` (with zenith
attenuation table). d2D and d3D clamped to >= 0.1 m. `sf_sigma` = scenpar `SF_sigma + SF_delta · lg(f)`, replaced
by `PL_SF`, `PL_sig1`, `PL_sig2`, `PL_sig3`, `PL_sig` where the model defines them. With an O2I track: d3D-in
removed from the distances (2D scaled by the same fraction) and the O2I loss added.

### 5.4 LOS probability (`set_scenario`)

Models used today: 36.873 UMi, UMa (height term for indoor UTs); 38.901 UMi, UMa, RMa, InH mixed, InH open,
InF SL/DL/SH/DH (r = 0.2/0.6/0.2/0.4, d_clutter = 10/2/10/2, h_c = 2/6 for SH/DH); 38.811 elevation tables
(Dense urban, Urban, Suburban = Rural); mmMAGIC UMi (= 38.901 UMi), mmMAGIC Indoor (= InH mixed);
QuaDRiGa Industrial `min(1, exp(-(d + 20)/80) · exp(min(hBS, 8)/5 - 1))`; QuaDRiGa UD2D (= UMi). The uniform
SOS draw is renormalised to a uniform distribution over the drop (empirical CDF), then LOS if `u < p_LOS`.

### 5.5 O2I (`gen_o2i_loss`)

Building type: uniform vs. low-loss fraction. d2D-in: inverse of the CDF of min(U1, U2) applied to one uniform
variable, times the maximum indoor distance; d3D-in from the geometry with a minimum outdoor distance of 10 m.
PL_in = 0.5 d2D-in; sigma_P 4.4 / 6.5 dB. All draws with one `SC_lambda`. mmMAGIC variant: alpha = 0.5 + U,
d_room = 2 + 2 U, PL_in = alpha · max(d2D-in - d_room, 0), elevation loss 20 · abs(theta / 90 deg), sigma_P =
4 + 0.1 f.


## 6. API

All declarations in `include/quadriga_gbsm.hpp`, namespace `quadriga_lib`, `double` only (D12), documentation
section "Geometry-based stochastic model" (D11). Positions are rows {x, y, z} in meters; heights are z
coordinates. `scenpar`, `pl_model`, `plpar` are the outputs of `conf_file_read` (D9); missing keys take their
default (`scenpar_validate(..., validate = false)` internally). `n_link` = number of links (rows of `rx_pos`),
`n_freq` = number of carrier frequencies.

The signatures below are the contract. The worker of each function may propose changes at review point A; the
manager accepts changes of names of internal details, and escalates changes of arguments, order or semantics
(D28).

Data flow for one builder (one BS site, one scenario):

```
lsf_init_phases(scenpar, phase, seed, &rx_pos, &tx_pos)        once per builder; the builder stores 'phase'
lsf_draw(scenpar, phase, tx_pos, rx_pos, rnd)                  normalized random values per link
path_loss(pl_model, plpar, scenpar, fc, tx_pos, rx_pos, pl, &sf_sigma, d3d_in, &rnd_he)    rnd_he = rnd.col(10)
lsp_generate(scenpar, fc, tx_pos, rx_pos, rnd, lsp, &sf_sigma, &abs_toa, ...)
```

Layout level (before builders exist): `los_probability` and `o2i_loss`. Their random inputs come from `sos_rand` /
`sos_randn` at the TR correlation distances; they are evaluated once per drop, so no phases are stored.

Every model function is deterministic: randomness enters only through normalized random values (N(0,1) or
U(0,1)) that the caller passes. Tests use exact oracles; the random fields have their own functions.

### 6.1 Moved functions

`conf_file_read`, `scenpar_validate`, `conf_file_list`, `conf_file_write`: declarations move unchanged from
`quadriga_tools.hpp` to `quadriga_gbsm.hpp`; their documentation blocks (C++, MEX, Python) change the section.
New numeric keys (6.9) are added to the key table; this changes `n_key` but not the interface.

### 6.2 LSF generator index

A fixed order, documented in the header and in each function:

| Index | Generator | Distribution | Decorrelation distance | Tx-side phases (reciprocity) |
|---|---|---|---|---|
| 0 | DS | Normal | `DS_lambda` | same as Rx |
| 1 | KF | Normal | `KF_lambda` | same |
| 2 | SF | Normal | `SF_lambda` | same |
| 3 | ASD | Normal | `AS_D_lambda` | Rx phases of ASA |
| 4 | ASA | Normal | `AS_A_lambda` | Rx phases of ASD |
| 5 | ESD | Normal | `ES_D_lambda` | Rx phases of ESA |
| 6 | ESA | Normal | `ES_A_lambda` | Rx phases of ESD |
| 7 | XPR | Normal | `XPR_lambda` | same |
| 8 | absolute TOA | Normal | `absTOA_lambda` | same |
| 9 | ground permittivity | Uniform | `SC_lambda` | same |
| 10 | UMa environment height hE | Uniform | `SF_lambda` (D19) | same |

A decorrelation distance of 0 is treated as 0.1 m, as in `sos_randn` (practically uncorrelated values, still
reproducible from the phases).

### 6.3 `lsf_init_phases`

```cpp
// Draw the random phases of the large-scale generators of one BS site and scenario
// - Generator order and reciprocity rules: see the LSF generator index
// - Bias removal over the positions in 'rx_pos' (and 'tx_pos' for dual mobility) when given and n_link > 2
void lsf_init_phases(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances)
                     arma::cube &phase,                                      // Output: Phases in [rad], [n_coeff, 11, 2]; slice 0 = Rx side, slice 1 = Tx side
                     arma::sword seed = -1,                                  // Seed, -1 = system random device
                     const arma::mat *rx_pos = nullptr,                      // Optional: Rx positions of the drop for the bias removal, [n_link, 3]
                     const arma::mat *tx_pos = nullptr,                      // Optional: Tx positions, [1, 3] or [n_link, 3]
                     bool dual_mobility = false,                             // Tx and Rx both move: the Tx side is a second device
                     bool remove_bias = true,                                // Remove the sample mean over the positions (D14)
                     const std::string &preset = "Comb256",                  // SOS preset, see sos_preset (D13)
                     bool horizontal = true);                                // true: 2D correlation in the x-y plane (TR); false: 3D (legacy)
```

### 6.4 `lsf_draw`

```cpp
// Spatially correlated normalized random values of the large-scale generators at a set of links
// - Columns 0 to 8 are Normal(0,1), columns 9 and 10 Uniform(0,1)
// - Dual mobility: ASD/ASA and ESD/ESA are coupled by the ACF at the Tx-Rx distance
// - Links with different floor IDs are uncorrelated; equal IDs share one field
void lsf_draw(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances)
              const arma::cube &phase,                                 // Phases from lsf_init_phases, [n_coeff, 11, 2]
              const arma::mat &tx_pos,                                 // Tx positions, [1, 3] or [n_link, 3]
              const arma::mat &rx_pos,                                 // Rx positions, [n_link, 3]
              arma::mat &rnd,                                          // Output: Random values, [n_link, 11]
              const arma::uvec *floor_id = nullptr,                    // Optional: Floor index per link, [n_link]; nullptr = all on one floor
              bool dual_mobility = false,                              // As in lsf_init_phases
              const std::string &preset = "Comb256",                   // Must match lsf_init_phases
              bool horizontal = true);                                 // Must match lsf_init_phases
```

### 6.5 `path_loss`

```cpp
// Path loss and shadow-fading standard deviation of a set of links
// - Models: logdist, logdist_simple, constant, winner_los, winner_nlos, winner_pathloss, dual_slope,
//   tripple_slope, nlos, satellite; parameters in 'plpar' as in the conf files (keys without "PL_")
// - An empty 'pl_model' gives zero loss
void path_loss(const std::string &pl_model,                                          // Path-loss model, value of 'PL_model'
               const std::unordered_map<std::string, std::vector<double>> &plpar,    // Path-loss parameters
               const std::unordered_map<std::string, double> &scenpar,               // Scenario parameters (SF_sigma, SF_delta)
               const arma::vec &fc,                                                  // Carrier frequencies in [Hz], [n_freq]
               const arma::mat &tx_pos,                                              // Tx positions, [1, 3] or [n_link, 3]
               const arma::mat &rx_pos,                                              // Rx positions, [n_link, 3]
               arma::mat &pl,                                                        // Output: Path loss in [dB], [n_link, n_freq]
               arma::mat *sf_sigma = nullptr,                                        // Optional output: SF standard deviation in [dB], [n_link, n_freq]
               const arma::vec *d3d_in = nullptr,                                    // Optional: Indoor part of the 3D distance in [m], [n_link]
               const arma::vec *rnd_he = nullptr,                                    // Optional: Uniform(0,1) per link for the random hE, column 10 of lsf_draw, [n_link]
               bool o2i_total_distance = true);                                      // O2I: PL_b with d3D-out + d3D-in (TR); false: outdoor distance only (legacy)
```

The O2I loss (from `o2i_loss`) is added by the caller; `path_loss` only uses `d3d_in` for the distances. With
`PL_hE_rand = 1` and `rnd_he` omitted, hE is the fixed `PL_hE` (documented).

### 6.6 `lsp_generate`

```cpp
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
                  double vs_ratio = 0.0,                                  // InF hall volume / surface in [m], overrides DS_VS (D26); 0 = use DS_VS
                  bool cholesky = true,                                   // true: Cholesky in TR order (L1); false: sqrtm (legacy)
                  bool clip_spreads = true);                              // Clip AS to 104 deg, ES to 52 deg (L2)
```

### 6.7 `los_probability`

```cpp
// LOS probability of a set of links, TR 38.901 Table 7.4.2-1 and other named models
// - Models: "38.901_RMa", "38.901_UMi", "38.901_UMa", "38.901_InH_Mixed", "38.901_InH_Open",
//   "38.901_InF_SL", "38.901_InF_DL", "38.901_InF_SH", "38.901_InF_DH", "38.901_InF_HH", "38.901_SMa",
//   "36.873_UMi", "36.873_UMa", "38.811_DenseUrban", "38.811_Urban", "38.811_Suburban", "38.811_Rural",
//   "QuaDRiGa_Industrial", "LOS" (always 1), "NLOS" (always 0)
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
                     arma::mat *los_soft = nullptr,     // Optional output: Soft LOS state in [0, 1], [n_link, n_freq]
                     bool indoor_ut = false);           // 36.873 UMa: height term for indoor UTs
```

Defaults of the InF models: the TR calibration values (S5): SL r = 0.2, d_clutter = 10, h_c = 2; DL r = 0.6,
d_clutter = 2, h_c = 6; SH as SL; DH as DL. The 38.811 elevation tables and the Industrial model are taken from
`set_scenario.m` (5.4). SMa: d_clutter = 30 m, heights 20 / 8 / 15 m, r = 2 % / 18 %, r_vegetation default 0.

### 6.8 `o2i_loss`

```cpp
// O2I building or car penetration loss of a set of UTs, TR 38.901 clause 7.4.3
// - Models: "low", "high", "low_A" (Table 7.4.3-2), "legacy_6GHz" (Table 7.4.3-3), "car", "car_metal" (7.4.3.2),
//   "mmMAGIC"
// - The model of each UT is chosen by comparing column 0 of 'rnd' with the cumulative 'weights'
void o2i_loss(const arma::vec &fc,                       // Carrier frequencies in [Hz], [n_freq]
              const arma::mat &rnd,                      // Random values per UT, [n_ut, n_rnd], see below
              arma::mat &loss,                           // Output: penetration loss in [dB], [n_ut, n_freq]
              const std::vector<std::string> &models,    // Models to mix, [n_model]
              const arma::vec &weights,                  // Probability of each model, sum 1, [n_model]
              double d2d_in_max = 25.0,                  // Maximum indoor 2D distance in [m] (key O2I_d2D_in_max)
              arma::vec *d2d_in = nullptr,               // Optional output: Indoor 2D distance in [m], 0 for car models, [n_ut]
              arma::uvec *model_index = nullptr,         // Optional output: Chosen model per UT, 0-based index into 'models', [n_ut]
              const arma::mat *tx_pos = nullptr,         // Optional: Tx positions for d3d_in and the mmMAGIC elevation, [n_tx, 3]
              const arma::mat *rx_pos = nullptr,         // Optional: UT positions, [n_ut, 3]
              arma::mat *d3d_in = nullptr,               // Optional output: Indoor part of the 3D distance in [m], [n_ut, n_tx]
              bool d2d_in_min_of_two = true);            // true: min of two uniforms (TR); false: legacy inverse CDF of one uniform
```

`rnd` columns (WP-6 fixes the exact list at review point A): U model choice, U d2D-in 1, U d2D-in 2, N sigma_P;
mmMAGIC adds U alpha, U d_room. The loss contains PL_tw + PL_in + sigma_P · N (car: mu + sigma · N). The mmMAGIC
elevation term needs `tx_pos` and `rx_pos` and then has one value per Tx: the worker proposes at review point A
whether `loss` becomes `[n_ut, n_freq, n_tx]` in that case or the elevation term moves to a separate output.

### 6.9 New conf keys

Numeric keys in the key table (defaults reproduce today's behaviour), added in WP-0:

| Key | Default | Class | Meaning | Placed in the v19 files |
|---|---|---|---|---|
| `LOS_state_lambda` | 0 | non-negative | Decorrelation distance of the LOS/NLOS state [m] (Table 7.6.3.1-2), 0 = uncorrelated | NLOS files: RMa 60, UMi 50, UMa 50, SMa 50, Indoor 10, InF SL/SH 5, DL/DH 1 |
| `IO_state_lambda` | 0 | non-negative | Decorrelation distance of the indoor/outdoor state [m], 0 = uncorrelated | O2I files: 50 |
| `O2I_d2D_in_max` | 0 | non-negative | Maximum indoor 2D distance [m], 0 = not stated | O2I files: UMa, UMi, SMa 25; RMa 10 |
| `DS_VS` | 0 | non-negative | Reference V/S ratio of the hall [m], 0 = no V/S term | InF files: LOS 4.27 (average of the two calibration halls, reproduces today's value), NLOS SL/DH 4, DL/SH 4.545 |
| `DS_VS_a` | 0 | non-negative | Factor a in lgDS mean = DS_mu + lg(a · V/S + b) | InF LOS 26, NLOS 30 (DS_mu = -9.35 / -9.44) |
| `DS_VS_b` | 0 | non-negative | Offset b | InF LOS 14, NLOS 32 |

The calibration halls (7.8.4): 120 x 60 x 10 m gives V/S = 4.0, 300 x 150 x 10 m gives V/S = 4.545.

New `PL_` keys (no interface change, `plpar` is open), added to the files in WP-3:

| Key | Model | Meaning | v19 files |
|---|---|---|---|
| `PL_hE_rand` | dual_slope, nlos | 1 = UMa random hE (TR note 1) from `rnd_he`; 0 = fixed `PL_hE` | UMa LOS, NLOS, both O2I |
| `PL_LOSmax` | nlos | 0 = no max with the LOS term; default 1 | SMa NLOS, SMa NLOS_O2I (replaces the zeroed LOS part) |
| `PL_Ax`, `PL_Bx`, `PL_Cx` | nlos | extra log-distance term included in the max | InF NLOS_DL (the InF-SL formula) |


## 7. Resolved review points of the API

Q1 to Q13 of the API draft were answered on 2026-10-23; the answers are D17 to D29. Nothing of the API is open.
Details that the workers settle at review point A (manager decides, D28): the exact `rnd` columns and the
mmMAGIC output shape of `o2i_loss` (6.8); the parameter names of the 38.811 tables inside `los_probability`;
whether `lsf_draw` applies the floor separation by an offset of a third coordinate or by per-floor phase offsets.


## 8. Implementation plan

### 8.1 Constraints (in every worker prompt)

- `md/SPEC_CPP_API.md` (C++), `md/SPEC_MATLAB_API.md` (MEX), `md/SPEC_PYTHON_API.md` (Python) are binding:
  test-first, stub, oracles from first principles, diagnostic switch `QD_TEST_VERBOSE` stays on, documentation
  blocks, validation before output, outputs resized only when the size differs.
- `double` only; C++17; must compile without warnings (`-Wall -Wextra -Wpedantic`) and with MSVC; OpenMP 2.0
  only; no BLAS/LAPACK (no `arma::chol`, `eig_sym`, `sqrtmat`, `norm`, `dot`, `solve`): the Cholesky and the
  Jacobi eigen-decomposition are written by hand.
- No `std::to_chars` for floating point (MATLAB R2024a, GLIBCXX <= 3.4.28). Every package that touches MEX code
  runs the gate `objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u`.
- Resources: at most 12 threads (`--parallel 12`, `OMP_NUM_THREADS <= 12`); never two builds or two test suites
  at the same time; check `cat /proc/loadavg` before heavy work; heavy tools (ASan, MATLAB, sweeps) once on the
  final code of a package. CUDA is not involved.
- quadriga-lib must not depend on QuaDRiGa-NG: no paths to it in sources or tests, no data derived from its files.
  NG and the TR are read-only until WP-15.
- Commit messages: no WP numbers, no references to `md/` documents (they are deleted at the end). The manager
  prepares the commits; Stephan commits (D31).
- Oracles: the TR formulas typed into the test file independently of the plpar/scenpar mapping. This tests the
  code and the v19 conf files at the same time: e.g. `path_loss` with the v19 UMa NLOS file must equal TR Table
  7.4.1-1 typed into the test.
- Internal header: `src/quadriga_gbsm_helpers.hpp` may be created for helpers used by more than one of the new
  source files (scenpar lookup with default, generator table, frequency scaling, the 8x8 roots). Approved
  exception to the "ask first" rule of the SPEC (manager default; Stephan may object at the first report).

### 8.2 Work packages

Sequential (one build at a time). WP-2 to WP-6 are independent of each other and may run in any order after WP-1.

**WP-0 Header move, new keys, Python submodule** (small)
- Create `include/quadriga_gbsm.hpp` (license, guards, the `static_assert`s of the other headers), move the four
  conf/scenpar declarations; `quadriga_lib.hpp` includes it; fix every include that relied on
  `quadriga_tools.hpp` for them (`src/`, `tests/`, `api_mex/`, `api_python/`).
- Section "Geometry-based stochastic model" in the four C++, MEX and Python documentation blocks; a
  `SECTION_DESC` block if the documentation tools use one for the other sections.
- Python: dispatcher `api_python/python_gbsm.cpp`, submodule `gbsm` registered in `python_main.cpp`, the four
  registrations moved from `python_tools.cpp`; Python tests call `quadriga_lib.gbsm.*`.
- Key table: the six numeric keys of 6.9 (with validation class, writer section and comment); key count tests in
  Catch2, MEX and Python updated (165 to 171 numeric keys, 172 MEX/Python fields).
- v19 files: values of the six keys as in 6.9, with TR references in the comments.
- SPEC files: header list and section names (`SPEC_CPP_API.md`), submodule table and dispatcher count
  (`SPEC_PYTHON_API.md`), section names (`SPEC_MATLAB_API.md`).
- Gate: full Catch2 suite, MEX suite (Octave and R2024a), Python suite.

**WP-1 Contracts and stubs** (small)
- The six declarations of section 6 in `quadriga_gbsm.hpp` under a group line `// ---- Large-scale fading ----`.
- Stubs in `src/lsf_init_phases.cpp`, `src/lsf_draw.cpp`, `src/path_loss.cpp`, `src/lsp_generate.cpp`,
  `src/los_probability.cpp`, `src/o2i_loss.cpp` per SPEC Phase 2 (sizes outputs, zeros, first documentation block).
- Review point A for all six contracts at once (manager).

**WP-2 Random fields: `lsf_init_phases`, `lsf_draw`** (medium)
- Generator table 6.2, frequency scaling per decorrelation distance, mirroring then bias removal (D21), 2D and 3D,
  floor separation, dual-mobility ASD/ASA and ESD/ESA coupling, lambda = 0 as 0.1 m.
- Tests: reciprocity (swap Tx and Rx positions gives DS, KF, SF, XPR, absTOA, GR, hE unchanged and ASD <-> ASA,
  ESD <-> ESA swapped); seed determinism; marginals (mean, std, uniform range) over a large grid; ACF estimate
  versus the preset at the scaled distance (tolerance from the SOS approximation error); bias removal makes the
  drop mean zero; 2D: values independent of z; floors: correlation between floors near 0; dual-mobility coupling
  equals the ACF at the Tx-Rx distance.

**WP-3 `path_loss`** (large)
- The 10 models of `get_pl.m` 1:1, the defaults of their missing keys, the clamps; `sf_sigma`; the new `PL_` keys
  (P1, P2, P3); O2I distances (P5) with the legacy option.
- v19 files: `PL_hE_rand`, `PL_LOSmax`, `PL_Ax/Bx/Cx` as in 6.9; comments with the optional NLOS formulas (D27).
- Tests: every TR Table 7.4.1-1 row (RMa, UMa, UMi, InH, InF x 5, SMa; LOS and NLOS) typed from the TR, compared
  with `path_loss` on the v19 files at several (d2D, hBS, hUT, fc) incl. both sides of the breakpoint; the UMa hE
  distribution (probability 1/(1 + C) of hE = 1 m and the discrete uniform set) over many `rnd_he` values; each
  legacy model against a hand-computed value; `sf_sigma` per breakpoint side.

**WP-4 `lsp_generate`** (large; the worker may split the commits into "mean and transform" and "extras")
- Mean and std (5.1) incl. `LSP_fc_min`, ESD terms, InF V/S (D26); cross-correlation: Cholesky in TR order and
  `sqrtm` by Jacobi; clipping; SF with `sf_sigma`; outputs `lsp_mu`, `lsp_sigma`; EoD offset; absolute TOA with
  the 2L/c bound; ground permittivity (5.5 of `gen_lsf_parameters`: dry / medium / wet interpolation).
- Tests: mean and std against the TR tables typed into the test for every v19 file at a reference point (incl.
  the fc clamp and the ESD height terms); EoD offset against `-mu_offset,ZOD` from Tables 7.5-7 to 7.5-12; sample
  covariance of the log-domain values equals the matrix of the file (both roots); Cholesky and sqrtm give the
  same covariance; `sqrtm` squared equals the matrix to 1e-12; clipping; multi-frequency uses the same x; absTOA
  rule (D22) and bound; permittivity at U = 0, 0.5, 1 equals the three ground types.

**WP-5 `los_probability`** (medium)
- All named models of 6.7 incl. soft LOS (7.6-18; `erfinv` written by hand or from the library if one exists).
- Tests: each model against the TR / `set_scenario.m` formula typed into the test at points on both sides of every
  breakpoint; limits (p = 1 at short distance, monotone in d2D where the model is); soft LOS: LOS_soft = 0.5 for
  G = -F(d), and the fraction of LOS_soft > 0.5 over many G equals p_LOS.

**WP-6 `o2i_loss`** (medium)
- Models of 6.8, model mixing by weights, d2D-in (min of two, legacy option), d3D-in geometry with the minimum
  outdoor distance of 10 m, mmMAGIC incl. elevation term.
- Tests: PL_tw of each model against Table 7.4.3-1/7.4.3-2 typed into the test; d2D-in distribution of min(U1,
  U2); model shares versus weights; d3D-in geometry on hand-built positions; car mean and std.

**WP-7 Conf files and documentation sweep** (small)
- All 23 v19 files read without ignored keys, validate, positive-definite correlation matrices (test exists from
  the parser port, extend it); comments reference the new functions; the release notes in `README.md` list the
  new functions, the header move and the Python submodule move.

**WP-8 MEX wrappers** (medium) — six wrappers per `md/SPEC_MATLAB_API.md`, MEX tests, R2024a gate.

**WP-9 Python wrappers** (medium) — six wrappers in `quadriga_lib.gbsm`, Python tests.

**WP-10 Review round 1** — a review worker reads the complete diff against this plan, the TR and the SPEC files
and writes findings (correctness first, then contract and documentation). It runs the suites once.
**WP-11 Fixes round 1.** **WP-12 Review round 2** (fresh reviewer, also checks the fixes). **WP-13 Fixes round 2.**
A third round only if round 2 finds correctness defects.

**WP-14 Merge into `main`** — on Stephan's request: merge in a temporary worktree, full suites, then
`git merge --ff-only` in the main worktree; `make documentation` and commit `html_docu/` afterwards.

**WP-15 Integration into QuaDRiGa-NG** — section 10.

### 8.3 Acceptance per package

- New Catch2 tests and the complete suite pass (`make test`), no warnings; for wrapper packages the MEX suite
  (Octave and R2024a with the objdump gate) and the Python suite pass.
- The documentation block describes what the function does, incl. the decisions of this plan.
- The worker's report follows SPEC review point C; the manager checks the diff, reruns the new tests and the
  Catch2 suite, and records the result in section 11.


## 9. Manager procedure (Phase 4)

The implementation manager is a separate session started by Stephan with the prompt in 9.2. It owns this document
from then on.

### 9.1 What the manager does

- For each work package: write the worker prompt (task, files, constraints of 8.1, the decisions that apply, the
  "manager defaults" for every open choice, repeated at the end so Stephan can change them before starting the
  worker); Stephan starts the worker session.
- Decide review points A, B, C of the workers within this plan (D28); escalate to Stephan only contract changes,
  deviations from a decision in section 3, and open TR questions. Record each decision as a new D-row.
- Review each result lightly: the diff, the new tests, the Catch2 suite; heavy runs (ASan, MATLAB, sweeps) only
  when the package touches them, otherwise taken from the worker report.
- Keep sections 4 (WP column), 8 and 11 current; update them before Stephan commits, so the status goes into the
  same commit.
- Prepare commits: themed cuts (one per function or function group, planning notes last), message files, the
  `git add` commands; fill in the hashes in section 11 afterwards.
- Short reports to Stephan per package with a clear recommendation.
- Merges into `main` only on request (WP-14). After WP-14, write the integration prompts (section 10).
- At the end: delete this document and the worker prompts (they go stale), as for the bug-fix series; the
  history keeps them.

### 9.2 Prompt for the manager session

```
You are the implementation manager for the port of the QuaDRiGa large-scale model to quadriga-lib
(branch 'fixes' in /sjc/quadriga-lib-fixes). Read md/gbsm_lsf_plan.md completely: it holds the decisions
(section 3), the API (section 6), the work packages (section 8) and your procedure (section 9). Then read
md/SPEC_CPP_API.md and skim md/SPEC_MATLAB_API.md and md/SPEC_PYTHON_API.md.

Start with WP-0: write the worker prompt into md/prompts/wp00_header_move.md (create md/prompts/), with
manager defaults for every open choice, and tell me when it is ready. Keep section 11 of the plan current.
Do not edit source files yourself unless a fix is a few lines and I agree. QuaDRiGa-NG and the 3GPP reference
are read-only until WP-15.
```


## 10. Hand-over for the QuaDRiGa-NG integration (WP-15)

Starts after WP-14. NG becomes writable. Split into prompts by the manager; suggested parts:

- **15a Library update in NG:** copy the new quadriga-lib (as in the earlier ports: `external/`, the MEX
  `+quadriga_lib`), copy the 23 v19 conf files to `quadriga_src/config`.
- **15b Builder LSF:** the builder stores `lsf_phase` (`[n_coeff, 11, 2]`) instead of the 8 + 2 `qd_sos` objects
  in `sos`, `absTOA_sos`, `gr_sos` (`init_sos` LSF part replaced by `lsf_init_phases`; `copy`, `split_rx`,
  `split_multi_freq`, `add_paths` adapted); `gen_lsf_parameters`, `get_sf_profile`, `get_lsp_map` become thin
  wrappers of `lsf_draw` + `path_loss` + `lsp_generate`; `lsp_vals` replaced by the `lsp_mu` / `lsp_sigma`
  outputs where it is used (`add_sdc`, `gen_ssf_parameters`); `get_pl` becomes a wrapper of `path_loss`;
  `get_channels` uses `sf_sigma` instead of `scale_sf` (D16). `simpar` gets the legacy switches of D4
  (Cholesky, clipping, 2D, O2I distance) with a "legacy" preset.
- **15c Layout:** `set_scenario` keeps a table group -> (LOS model name, member files, parameters) and calls
  `los_probability`; LOS and indoor state distances from `LOS_state_lambda`, `IO_state_lambda`; groups for the v19
  set incl. SMa and InF-HH; `gen_o2i_loss` replaced by `o2i_loss` and deleted; an in-car option for UTs.
- **15d Parity test (D23):** a MATLAB script computes, with the pre-integration code, PL, SF and the 8 LSPs of the
  legacy files for single mobility at fixed phases and stores them in NG's test data; the test compares the new
  code with legacy flags to 1e-12 relative.
- **15e NG unit tests and tutorials:** all `xunit_new` tests pass in MATLAB R2024a and Octave; the calibration
  tutorials `calib_3GPP_38901_large_scale.m` and `calib_3GPP_38901_full.m` run with the v19 groups.
- Retired MATLAB code (target): `gen_o2i_loss.m`; the LSF part of `init_sos.m`; most of `gen_lsf_parameters.m`,
  `get_pl.m`, `get_sf_profile.m`, `get_lsp_map.m`; the p_LOS formulas in `set_scenario.m`; the `lsp_vals`
  property.


## 11. Status

| WP | Content | State | Commit(s) |
|---|---|---|---|
| WP-0 | Header move, new keys, Python submodule | open | |
| WP-1 | Contracts and stubs | open | |
| WP-2 | `lsf_init_phases`, `lsf_draw` | open | |
| WP-3 | `path_loss` | open | |
| WP-4 | `lsp_generate` | open | |
| WP-5 | `los_probability` | open | |
| WP-6 | `o2i_loss` | open | |
| WP-7 | Conf files and documentation sweep | open | |
| WP-8 | MEX wrappers | open | |
| WP-9 | Python wrappers | open | |
| WP-10 | Review round 1 | open | |
| WP-11 | Fixes round 1 | open | |
| WP-12 | Review round 2 | open | |
| WP-13 | Fixes round 2 | open | |
| WP-14 | Merge into main | open | |
| WP-15 | Integration into QuaDRiGa-NG | open | |
