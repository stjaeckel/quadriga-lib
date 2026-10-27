# Spec: port of the QuaDRiGa large-scale model to quadriga-lib

Status: final for implementation (2026-10-26). Branch `fixes` of `/home/stephan/sjc/quadriga-lib-fixes`, base cebd0c0 (v0.12.3).

This spec is self-contained. Together with the source code of quadriga-lib and QuaDRiGa-NG and the published
material it cites (3GPP TR 38.901 v19.5.0 above all), it contains everything an implementation needs. Every
requirement is stated where it applies, with its reason where the choice is not obvious.

No section number of this spec, no `md/` path, no local path and no QuaDRiGa-NG line number may appear in any
produced source, header, test, wrapper, conf file or documentation block (9.1).

Process: the review points A, B and C of the repository's API conventions `md/SPEC_CPP_API.md`,
`md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md` are **not** stops for a human. The implementation agent decides
within this spec, never against it, and records every decision and every deviation in its run report (section 7).
Where this spec and a SPEC file disagree, this spec wins; the known differences are listed in 1.4. The three SPEC
files are permanent coding-rule files of the repository, not working documents of this series, and they are
binding for the implementation.

Contents

1. Scope and requirements
2. Public API (`include/quadriga_gbsm.hpp`)
3. Normative behaviour (shared definitions, the six new functions, the moved functions)
4. Conf-key changes and the 23 v19 files
5. Catch2 tests
6. MEX and Python wrappers and their tests
7. Implementation runs: Run 1, Run 2a, Run 2b, Run 3a, Run 3b, Run 4a, Run 4b
8. Requirements from the later SSF port
9. Constraints (copied into every run)
10. Integration hand-over (QuaDRiGa-NG, after the merge)
11. Appendix: tables


## 1. Scope and requirements

### 1.1 Goal

Port the large-scale part of QuaDRiGa (QuaDRiGa-NG `quadriga_src`) to the C++ core of quadriga-lib, with MEX and
Python wrappers:

- path loss and shadow-fading standard deviation (`qd_builder.get_pl`),
- the large-scale parameters (LSPs) DS, KF, SF, ASD, ASA, ESD, ESA, XPR (`qd_builder.lsp_vals`, the LSF part of
  `init_sos`, `gen_lsf_parameters`, `get_sf_profile`, `get_lsp_map`),
- the quantities drawn together with them: absolute TOA offset, ground-reflection permittivity, EoD offset,
- LOS probability and soft LOS state (probability part of `qd_layout.set_scenario`),
- O2I penetration loss of TR 38.901 (the TR part of `qd_layout.gen_o2i_loss`).

Goal 1: full coverage of the large-scale model of 3GPP TR 38.901 v19.5.0 (clauses 7.4, 7.5 steps 2 to 4, 7.6.3.3,
7.6.9 and the LSF-related parts of 7.6.5, 7.6.10). Goal 2: QuaDRiGa-NG replaces its MATLAB implementations by
library calls (section 10).

Ten functions in the new public header `include/quadriga_gbsm.hpp`: four moved ones (`conf_file_read`,
`scenpar_validate`, `conf_file_list`, `conf_file_write`) and six new ones (`lsf_init_phases`, `lsf_draw`,
`path_loss`, `lsp_generate`, `los_probability`, `o2i_loss`).

Data flow for one builder (one BS site, one scenario):

```
conf_file_read(fn, scenpar, ..., &pl_model, &plpar)                     once per scenario file
lsf_init_phases(scenpar, phase, seed, &tx_pos, &rx_pos)                once per builder; the builder stores 'phase'
lsf_draw(scenpar, phase, tx_pos, rx_pos, rnd, &floor_id)               normalized random values per link, [n_link, 11]
path_loss(pl_model, plpar, scenpar, fc, tx_pos, rx_pos, pl,
          &sf_sigma, nullptr, &rnd_he)                                  rnd_he = rnd.col(10); d2d_in only for legacy
lsp_generate(scenpar, fc, tx_pos, rx_pos, rnd, lsp, &sf_sigma, ...)    final LSPs per link and frequency
```

Layout level (before builders exist), in this order per drop:

```
o2i_loss(fc, rnd_o2i, models, weights, loss, d2d_in_max, &d2d_in)      indoor UTs: penetration loss and d2D-in
d2d_out = max(d2D - d2d_in, 0)                                          caller; outdoor UTs: d2d_in = 0
los_probability(model, d2d_out, h_bs, h_ut, p_los, ...)                 TR Table 7.4.2-1 uses d2D-out
```

`o2i_loss` runs before `los_probability` because Table 7.4.2-1 evaluates the LOS probability of an O2I UT at
d2D-out = d2D - d2D-in; the caller clamps d2D-out at 0 (`los_probability` rejects negative distances). Their random
inputs come from `sos_rand` / `sos_randn` at the TR correlation distances (Table 7.6.3.1-2, clause 7.6.3.3); they
are evaluated once per drop, so no phases are stored. Every model function is deterministic: randomness enters
only through normalized random values (N(0,1) or U(0,1)) passed by the caller. The layout also derives the floor
index of each UT, which the builder passes to `lsf_draw` as `floor_id` (section 10.3).

### 1.2 Requirements (binding)

Each requirement is binding for the implementation; the sections named in parentheses give the details.

Scope and policy

- Besides path loss, SF standard deviation and the 8 LSPs, the library covers the LOS probability and soft LOS
  state, the O2I penetration loss, the absolute TOA, the ground-reflection permittivity, the EoD offset and the v19
  ESD terms. Reason: full coverage of the TR large-scale model.
- Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default. QuaDRiGa's behaviour is available through
  an argument where QuaDRiGa-NG needs it for its legacy mode: 3D correlation (`horizontal = false`), symmetric
  square root of the LSP correlation matrix (`cholesky = false`), no spread clipping (`clip_spreads = false`),
  outdoor path-loss distance (`d2d_in` of `path_loss`). Reason: TR results by default, today's QuaDRiGa values
  reproducible where they are still wanted.
- Where QuaDRiGa's behaviour is a bug, the library implements the corrected rule without a legacy option:
  `winner_los` uses `C2` beyond the breakpoint and the breakpoint from the heights of each link (3.3.4); each
  Normal field is corrected once over all positions where it is evaluated, and the Tx side is mirrored from the
  corrected phases (3.1); in dual mobility `lsf_draw` applies no extra 2x2 coupling to ASD/ASA and ESD/ESA (3.2
  step 5); a standard deviation that the distance, height or elevation terms make negative is limited at 0
  (3.0.4). Reason: a legacy option would preserve the error.
- `dual_slope` evaluates the loss beyond the breakpoint with the 3D breakpoint distance for every scenario, also for
  RMa and SMa LOS, where Table 7.4.1-1 uses the 2D breakpoint distance. Reason: the difference is small (bound in
  3.3.4) and does not justify a key and a second branch.
- Path loss: the 10 parametric models of `get_pl.m` with their formulas (3.3.4); TR behaviours they lack become new
  `PL_` keys (4.3). The optional NLOS formulas of Table 7.4.1-1 are documented in comments of the v19 NLOS files
  only, without new files (4.4.2).
- O2I penetration loss: only the models of TR 38.901 clause 7.4.3 (`low`, `high`, `low_A`, `legacy_6GHz`, `car`,
  `car_metal`) and the TR rule for the indoor distance (minimum of two uniforms) (3.6). Reason: the mmMAGIC model,
  the TR v14 wall values and QuaDRiGa's legacy d2D-in distribution are deprecated and stay in QuaDRiGa-NG if it
  still needs them; QuaDRiGa's O2I losses are not reproducible anyway, because it draws the deviation from a
  uniform instead of a Normal variable.
- LOS probability: named models in C++ (3.5); model parameters (InF and SMa clutter geometry) are function
  arguments with the TR values as defaults. The table of scenario groups (group name to model and member files)
  stays with the caller.
- The renormalisation of the LOS-state draw that QuaDRiGa's `set_scenario.m` applies is not part of the library.

Interfaces

- New public header `include/quadriga_gbsm.hpp` (2.1) with the six new functions; `conf_file_read`,
  `scenpar_validate`, `conf_file_list`, `conf_file_write` move there unchanged from `quadriga_tools.hpp` (clean
  move); `quadriga_lib.hpp` includes the new header. Later functions of the model (SSF) go there too.
- Documentation section "Geometry-based stochastic model" for the six new and the four moved functions (they
  leave "Channel generation functions").
- Function names `lsf_init_phases`, `lsf_draw`, `path_loss`, `lsp_generate`, `los_probability`, `o2i_loss`; free
  functions. The random state is a plain phase array in and out: the caller stores it and re-evaluates along tracks
  and on maps with it. Reason: no hidden state, the builder keeps plain arrays.
- Inputs `scenpar` and `plpar` exactly as `conf_file_read` returns them; missing scenpar keys get their defaults.
- `double` only, no templates.
- Output units as QuaDRiGa's builder: DS in s, KF, SF and XPR as linear power ratios, AS and ES in degrees, PL in
  dB. Arguments in degrees carry `_deg` in their name or say so in the comment (approved exception to
  SPEC_CPP_API).
- LSP output: one cube `[n_link, n_freq, 8]`, slices DS, KF, SF, ASD, ASA, ESD, ESA, XPR.
- Dual mobility is an explicit argument `bool dual_mobility`.
- Python: new submodule `quadriga_lib.gbsm` for the ten functions; the four conf functions move there from
  `quadriga_lib.tools` (breaking for Python users). MATLAB names stay flat and unchanged.
- Conf format fixed: no new file type, no new string keys, no change of the reader and writer interfaces. New
  numeric keys in the key table and new `PL_` keys are allowed. Numeric group-level settings (LOS-state and
  indoor-state decorrelation distances, maximum indoor distance of O2I UTs) are scenpar keys in the files where
  they are unique (NLOS files, O2I files) (4.1).
- Precision of conf values: a value in a conf file is written with the precision of its source, as TR 38.901
  prints it (`31.84`, `-9.35`, `43.42`). A derived value (computed from a TR formula, a hall geometry or a
  constant such as `c = 3e8`: `40/3`, `20·pi/3`, V/S, the RMa/SMa constants for given h and W) is written with at
  most 4 significant digits, fewer where they suffice, unless 4 digits change a result by more than 0.01 dB (path
  loss, SF) or 0.1 % (LSP means, distances) over the TR ranges; then with the fewest digits that keep the change
  within these limits. The exact expression may stand in the comment (`47/11 = 4.27`). Tests against TR oracles
  take their tolerance from the bound of this rounding, stated in a comment (5.3, 5.4). Reason: long numbers such
  as `4.2727272727272725` make the files unreadable, and the change is far below the model uncertainty (shadow
  fading 4 to 8 dB, LSP standard deviations of 0.1 to 0.5 in lg); bit-identical results are not required.

Model rules

- Spatial correlation of the LSF generators in the horizontal plane (2D) plus an optional floor ID; different
  floors are uncorrelated (Table 7.5-6 note 3). 3D correlation is the legacy option (3.1, 3.2). The floor ID is
  the caller's floor index of the UT (10.3).
- ACF of the LSF generators: SOS preset `Comb256` by default (QuaDRiGa); the preset is an argument, `Exp256` gives
  the exponential ACF of the TR (eq. 7.4-5; clause 7.6.3.1). Reason: the TR gives the exponential ACF "with
  sufficient accuracy" as a fit, not as an exact requirement; its kink at zero distance makes the values evolve
  roughly along mobility tracks. `Comb256` (Gaussian near zero, exponential tail, same decorrelation distance)
  gives smooth evolution. A caller who wants the literal ACF of the TR passes `"Exp256"`. The SOS functions of the
  library (`sos_randn`, `sos_rand`, `sos_randi`) keep their default `Comb256` as well.
- Bias removal of the LSF phases: optional, on by default whenever drop positions are passed. Every Normal field is
  corrected once over the positions where it is evaluated, then the Tx side is set by mirroring (3.1). Reason: zero
  drop mean with exact mirror relations and exact swap reciprocity.
- UMa random environment height hE (Table 7.4.1-1 note 1): a spatially consistent uniform variable, generator 10 of
  the LSF phase set, decorrelation distance `SF_lambda`, no new key (3.3.6).
- SF standard deviation: `path_loss` returns `sf_sigma` per link and frequency; `lsp_generate` takes it as optional
  input and returns the final SF (3.3.3, 3.4).
- Root of the LSP cross-correlation matrix: Cholesky in the TR order of clause 7.5 step 4 by default; the
  symmetric square root as option (3.4.3). Reason for the option: the symmetric root does not depend on the order
  of the parameters, and it is QuaDRiGa's rule.
- Absolute TOA (clause 7.6.9): active if `absTOA_mu > -30`, else 0; bounded by `2·absTOA_Lmax/c` when
  `absTOA_Lmax > 0` (3.4.2). The InF and InH NLOS files set `absTOA_Lmax` to the largest dimension of the
  calibration hall or office of clause 7.8 (4.4.1), as they set `DS_VS` from the same halls; a caller with another
  hall sets the key in its scenpar.
- InF delay spread (Table 7.5-6 Part-3): keys `DS_VS_a`, `DS_VS_b`, `DS_VS` in the InF files; a caller with
  another hall sets `DS_VS` in its scenpar copy, as it does for `absTOA_Lmax` (3.0.4, 4.1). Reason: an argument
  that overrides the key would duplicate it.
- Legacy parity: with the legacy arguments and the same random values, `path_loss` and `lsp_generate` reproduce
  today's QuaDRiGa values to 1e-12 relative in single mobility, except where this spec corrects QuaDRiGa and where
  QuaDRiGa evaluates per track segment what the library evaluates per link (both listed in 10.2; the comparison
  rules are in 10.4). This is checked in the integration with reference values from a MATLAB script in
  QuaDRiGa-NG; nothing in quadriga-lib depends on QuaDRiGa-NG.

Conventions of the existing conf parser that stay valid: elevation convention (QuaDRiGa names ESD, ESA, EoD;
`elevation = 90 deg - zenith`; the EoD offset has the opposite sign of the TR's ZOD offset); the formulas of the 27
v19 keys (3.0.4); the v19 conf files live in `data/config/` under Apache-2.0.

### 1.3 Out of scope

- Small-scale fading (SSF): cluster-number variability (7.6.15), near field (7.6.13), spatial non-stationarity
  (7.6.14), polarization variability (7.6.16), soft-LOS channel combination (7.6-19), clusters, rays, delays,
  angles. The keys exist already; section 8 lists what the SSF port needs from this one.
- Blockage (7.6.4), ISAC (7.9), map-based model (clause 8).
- LOS-state and indoor-state draws themselves (a `sos_rand` call of the caller), building-type draws, spatially
  consistent O2I variables (callers use `sos_rand` / `sos_randn` at the TR distances).
- The mmMAGIC O2I model, the TR v14 high-loss wall and QuaDRiGa's legacy d2D-in distribution (they stay in
  QuaDRiGa-NG if it still needs them).
- Changes in QuaDRiGa-NG (section 10) and the merge into `main`.
- The generated documentation in `html_docu/` (rebuilt after the merge by `make documentation`).

### 1.4 Differences to the SPEC files (approved)

- `double` only, no templates, no explicit instantiations, no float/double parity tests.
- Degrees and linear power ratios in outputs (1.2).
- The argument order of 2.1 (outputs before optional inputs) is the contract.
- Catch2 diagnostics via `qd_test_diag.hpp`, area `QD_DIAG_GBSM` (5.0).
- Review points are not stops (see the header).
- New internal header `src/quadriga_gbsm_helpers.hpp`, limited to helpers used by more than one source file
  (3.0.5); the phase-set machinery is file-local in `src/qd_gbsm_phases.cpp`.
- Each implementation run adds to `include/quadriga_gbsm.hpp` only the declarations it defines (section 7), so
  that no intermediate state declares functions without definition.


## 2. Public API

### 2.1 Header `include/quadriga_gbsm.hpp` (final text)

The final content of the file is given below. Run 1 creates it with the include guard, the includes, the
`static_assert` lines, the "Scenario parameters" group and the comment block of the "Large-scale fading" group
(generator index); the four moved declarations are copied unchanged, including their comments, from
`include/quadriga_tools.hpp`: the block from the line after the group line `// ---- Scenario parameters ----` to
the end of the `conf_file_write` declaration (its last argument `bool write_defaults = false);` with its comment).
The block and the group line are removed there. Each later run adds the declarations it defines, at their place in this text:
Run 2a `lsf_init_phases`, `lsf_draw`; Run 2b `lsp_generate`; Run 3a `path_loss`; Run 3b `los_probability`,
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
    // - The functions are documented in src/qd_gbsm_phases.cpp (lsf_init_phases, lsf_draw), src/path_loss.cpp,
    //   src/lsp_generate.cpp, src/los_probability.cpp, src/o2i_loss.cpp

    // Draw the random phases of the large-scale generators of one BS site and scenario
    // - Generator order and reciprocity rules: see the LSF generator index above
    // - Bias removal over the positions in 'rx_pos' (and 'tx_pos' for dual mobility) when given and n_link > 2
    void lsf_init_phases(const std::unordered_map<std::string, double> &scenpar, // Scenario parameters (decorrelation distances)
                         arma::cube &phase,                                      // Output: Phases in [rad], [n_coeff, 11, 2]; slice 0 = Rx side, slice 1 = Tx side
                         arma::sword seed = -1,                                  // Seed, -1 = system random device
                         const arma::mat *tx_pos = nullptr,                      // Optional: Tx positions, [1, 3] or [n_link, 3]
                         const arma::mat *rx_pos = nullptr,                      // Optional: Rx positions of the drop (one floor) for the bias removal, [n_link, 3]
                         bool dual_mobility = false,                             // Tx and Rx both move: the Tx side is a second device
                         bool remove_bias = true,                                // Remove the sample mean over the positions
                         const std::string &preset = "Comb256",                  // SOS preset, see sos_preset
                         bool horizontal = true);                                // true: 2D correlation in the x-y plane (TR); false: 3D (legacy)

    // Spatially correlated normalized random values of the large-scale generators at a set of links
    // - Columns 0 to 8 are Normal(0,1), columns 9 and 10 Uniform(0,1)
    // - Dual mobility: the Tx side uses the mirror generator, so ASD/ASA and ESD/ESA are correlated by the ACF
    //   at the Tx-Rx distance
    // - Links with different floor IDs are uncorrelated at equal horizontal positions; equal IDs share one field
    // - Dual mobility: both devices of a link are on the link's floor
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
                   const arma::vec *d2d_in = nullptr,                                    // Optional: Indoor 2D distance in [m], [n_link]; given = outdoor distance only (legacy); 0 = outdoor link, geometric distance
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
                      bool cholesky = true,                                   // true: Cholesky in TR order; false: symmetric square root (legacy)
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
    // - Models: "low", "high", "low_A" (Table 7.4.3-2), "legacy_6GHz" (Table 7.4.3-3), "car", "car_metal" (7.4.3.2)
    // - The model of each UT is chosen by comparing column 0 of 'rnd' with the cumulative 'weights'
    void o2i_loss(const arma::vec &fc,                       // Carrier frequencies in [Hz], [n_freq]
                  const arma::mat &rnd,                      // Random values per UT, [n_ut, 4]: U model, U d2D-in 1, U d2D-in 2, N deviation
                  const std::vector<std::string> &models,    // Models to mix, [n_model]
                  const arma::vec &weights,                  // Probability of each model, sum 1, [n_model]
                  arma::mat &loss,                           // Output: Penetration loss in [dB], [n_ut, n_freq]
                  double d2d_in_max = 25.0,                  // Maximum indoor 2D distance in [m] (key O2I_d2D_in_max)
                  arma::vec *d2d_in = nullptr,               // Optional output: Indoor 2D distance in [m], 0 for car models, [n_ut]
                  arma::uvec *model_index = nullptr);        // Optional output: Chosen model per UT, 0-based index into 'models', [n_ut]
}

#endif
```

The four moved declarations are inserted verbatim at the comment "(conf_file_read, ...: moved unchanged ...)",
replacing that comment line.

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
| 9 | ground permittivity | Uniform | `SC_lambda` | 9 | no |
| 10 | UMa environment height hE | Uniform | `SF_lambda` | 10 | no |

A decorrelation distance of 0 is used as 0.1 m (as `sos_randn`). The generator of column 8 exists also when the
absolute TOA is disabled (`absTOA_mu <= -30`); `lsp_generate` then ignores it. The bias removal applies to the
Normal generators 0 to 8 only; the Uniform generators 9 and 10 are not corrected.

### 2.3 Interface choices

| Item | Rule |
|---|---|
| `rnd` columns of `o2i_loss` | 4 columns: 0 U model choice, 1 U d2D-in #1, 2 U d2D-in #2, 3 N(0,1) deviation (sigma_P, car). |
| 38.811 table handling | Fixed tables inside `los_probability` (Appendix 11.2), linear interpolation over the elevation, no parameters; `d_clutter`, `r_clutter`, `h_clutter` are ignored by these models. The elevation is computed from the absolute BS height, `atan2(h_bs, d2d_out)`. |
| Model names `LOS`, `NLOS` | Kept (p = 1 / 0), so that the caller's group table is uniform. |
| Floor separation method | Per-floor phase offsets, not a coordinate offset (3.2.1). Reason: works in 2D and 3D mode, keeps the mirror relations, needs no large coordinate offsets (which would push small decorrelation distances towards the precision limit of the kernels), and gives exactly zero correlation at zero horizontal lag (UTs stacked above each other, the most relevant case) within a block of `n_coeff` floor IDs; the residual correlation at other lags is stated in 3.2.1. |


## 3. Normative behaviour

### 3.0 Shared definitions

#### 3.0.1 Notation and conventions

- `lg(x) = log10(x)`, `ln` the natural logarithm. Angles in formulas are in radians unless the name ends in `_deg`
  or the text says degrees. `atan` is the one-argument arctangent, `atan2(y, x)` the two-argument one.
- `f = fc / 1e9` is the carrier frequency in GHz; `fc` is always given in Hz. `n_freq = fc.n_elem`.
- `c = 299792458` m/s wherever the library computes with the speed of light. The TR breakpoint formulas use
  `c = 3e8`; that value is part of the conf-file constants (`PL_E = 4/0.3 = 40/3` and `2·pi/0.3 = 20·pi/3`,
  written as 13.33 and 20.94, 1.2), not of the code.
- Positions are rows `{x, y, z}` in meters. `n_link = rx_pos.n_rows`. `tx_pos` has 1 row (used for every link) or
  `n_link` rows; `tx(l)` denotes row 0 or row `l`.
- Link geometry: `dx = rx(l) - tx(l)`; `d2D = sqrt(dx_x² + dx_y²)`; `d3D = sqrt(d2D² + dx_z²)`; `hBS = tx(l).z`;
  `hUT = rx(l).z`. All heights are z coordinates (no ground model).
- `n_link = 0` is valid in all new functions: `rx_pos` is `[0, 3]`, outputs get 0 rows. `tx_pos` may then
  have 0 or 1 rows.
- Optional outputs (`nullptr`) are neither allocated nor computed. Outputs are resized only if their size differs,
  then every element is written (SPEC_CPP_API "Outputs"). All validation runs before any output is touched.
- Exceptions: `std::invalid_argument` for invalid input; messages follow SPEC_CPP_API (full sentence, argument in
  single quotes). Exceptions thrown by called library functions (`scenpar_validate`, ...) pass through unchanged.
- Precision: all functions compute in double. The only single-precision part is the sine/cosine evaluation
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
| a non-finite entry in `rx_pos` / `tx_pos` | `Input 'rx_pos' must be finite.` / `Input 'tx_pos' must be finite.` |
| `fc` empty (functions that take `fc`) | `Input 'fc' cannot be empty.` |
| an entry of `fc` not finite or `<= 0` | `Input 'fc' must be finite and > 0.` |
| optional vector input with wrong length | `Number of elements in '<name>' does not match number of rows in 'rx_pos'.` |

#### 3.0.3 Scenario parameters inside the functions

Each function that takes `scenpar` completes and validates a copy with `scenpar_validate(copy, "legacy",
&lsp_xcorr_or_nullptr, nullptr, true)`: missing keys get their default; an unknown key throws
(`Unknown key 'X' in 'scenpar'.`); every value is checked by its validation class (e.g.
`Value of 'XPR_lambda' must be finite and >= 0.`), and the cross-key rules of the key table apply (incl. the
`DS_VS` rule of 4.2). These messages pass through unchanged; the functions have no own key checks. Consequence
(accepted): an invalid value of a key that the function does not use throws as well; a `scenpar` from
`conf_file_read` or `scenpar_validate` never contains one. The return value of `scenpar_validate` (positive
definite) is used by `lsp_generate` only.

#### 3.0.4 LSP mean and standard deviation (shared by `lsp_generate` and the default `sf_sigma` of `path_loss`)

Source: `qd_builder.m` `get.lsp_vals` (lines 233 to 309), `gen_lsf_parameters.m` lines 232 to 268; the terms of the
v19 keys are given below. LSP index `i = 0..7` with key prefix `P_i = DS, KF, SF, AS_D, AS_A, ES_D, ES_A, XPR`.

Per link `l` and frequency `k`:

```
d   = max(d2D, 1e-4)                    clamped 2D distance [m]
h   = max(hBS, 1e-4)                    clamped BS height [m], used in all terms (also ES_D_mu_C, ES_D_mu_D)
a   = atan(h / d)                       elevation of the BS seen from the UT [rad]
f'  = max(f, LSP_fc_min)                frequency for all LSP terms [GHz]; LSP_fc_min = 0 gives f' = f

mu_i    = P_mu + P_gamma · lg(P_omega + f') + P_epsilon · lg(d) + P_zeta · lg(h) + P_alpha · lg(a)
s_i     = max(0, P_sigma + P_delta · lg(P_omega + f'))
sigma_i = max(0, s_i + P_kappa · lg(d) + P_tau · lg(h) + P_beta · lg(a))
```

- The inner limit is QuaDRiGa's (`sigma(sigma < 0) = 0` in `get.lsp_vals`). The outer limit is the library's:
  QuaDRiGa adds the distance, height and elevation terms after its limit, so the std can become negative (e.g. a
  QuaDRiGa-NG satellite file with `SF_sigma = 1.4`, `SF_beta = 1` below 2.3 deg elevation), which flips the sign
  of the realization and the sign of its cross-correlations. A standard deviation is never negative, so the
  library limits the complete value at 0. The v19 files have no `_kappa`, `_tau` or `_beta` terms, so the limit
  never acts for them. The default `sf_sigma` of `path_loss` (3.3.3) is therefore always >= 0, as `lsp_generate`
  requires.

- SF (`i = 2`) has no keys `SF_mu`, `SF_gamma`, `SF_epsilon`, `SF_zeta`, `SF_alpha`: they are 0, so `mu_2 = 0`.
  `SF_omega`, `SF_sigma`, `SF_delta`, `SF_kappa`, `SF_tau`, `SF_beta` exist.
- DS (`i = 0`), V/S term of Table 7.5-6 Part-3, active only if `DS_VS > 0`:
  `mu_0 += lg(DS_VS_a · DS_VS + DS_VS_b)`.
- ESD (`i = 5`), v19 terms, applied after the general formula and before the floor:
  `mu_5 += ES_D_mu_A · d / 1000 + ES_D_mu_B · hUT + ES_D_mu_C · abs(hUT - h) + ES_D_mu_D · max(hUT - h, 0)`, then
  `mu_5 = max(mu_5, ES_D_mu_min)` (`ES_D_mu_min` defaults to `-Inf`). `hUT` is the raw z of the Rx.
- Units: DS mean and std in lg(s); ASD, ASA, ESD, ESA in lg(deg); KF, SF, XPR in dB.
- The same `(l, k)` values serve `lsp_mu`, `lsp_sigma` and the transformation in 3.4.
- Frequency terms: a term with `P_gamma = 0` (`P_delta = 0`) contributes 0 to `mu_i` (`s_i`). An active term
  (`P_gamma` or `P_delta` not 0) with `P_omega + f' <= 0` (NaN included) throws, before any output is written:
  `Scenario parameter '<P>_omega' plus the frequency in GHz must be > 0 where '<P>_gamma' or '<P>_delta' is not 0.`
  (`P_omega` accepts any value; without the check the mean would be NaN and the NaN of the std would disappear in
  the limit at 0). `lsp_generate` checks all 8 LSPs, `path_loss` only SF and only when `sf_sigma` is requested.

#### 3.0.5 Internal header `src/quadriga_gbsm_helpers.hpp` and the phase-set file `src/qd_gbsm_phases.cpp`

The internal header is included by the source files named below and by nothing public. It holds only `static
inline` helpers (and one class) used by more than one source file; names start with `qd_gbsm_`. It starts with a
documentation block (comment lines) that states: purpose (helpers shared by the source files of the geometry-based
stochastic model, not part of the public API), the users of each helper, the LSP order and the units. Content:

```cpp
// Common validation of the link positions (3.0.2 rows 1 to 4); nullptr skips the checks of that argument
//                                                    users: qd_gbsm_phases.cpp, path_loss.cpp, lsp_generate.cpp
static inline void qd_gbsm_check_links(const arma::mat *tx_pos, const arma::mat *rx_pos);

// Validation of the carrier frequencies (3.0.2 rows 5 and 6)
//                                       users: path_loss.cpp, lsp_generate.cpp, los_probability.cpp, o2i_loss.cpp
static inline void qd_gbsm_check_fc(const arma::vec &fc);

// LSP mean and std per link and frequency of a completed scenpar (3.0.4)    users: lsp_generate.cpp, path_loss.cpp
// - The constructor reads the coefficients and evaluates the frequency terms at 'fc' for the LSPs in 'lsps'
//   (bit i = LSP i) and throws for an invalid frequency term (3.0.4); set_link() takes the geometry of one link,
//   get() returns mean and std of one LSP and frequency. No [n_link, n_freq, 8] intermediate is formed, so the
//   default sf_sigma of path_loss costs only the SF std
class qd_gbsm_lsp_stats
{
public:
    qd_gbsm_lsp_stats(const std::unordered_map<std::string, double> &sp, const arma::vec &fc, unsigned lsps = 0xFFu);
    void set_link(double d2D, double hBS, double hUT);
    void get(arma::uword i, arma::uword k, double &mu, double &sigma) const;
};

// Cholesky-Banachiewicz of a symmetric 8x8 matrix, column-major double[64]: A = L·L^T, L lower triangular with
// zeros above the diagonal; returns false at the first pivot that is not > 0 (NaN included)
//                                                                          users: lsp_generate.cpp, qd_scenpar.cpp
static inline bool qd_gbsm_chol8(const double *A, double *L);
```

`qd_scenpar_xcorr` in `src/qd_scenpar.cpp` calls `qd_gbsm_chol8` (the library has one 8x8 Cholesky). The linear
table interpolation `qd_interp_linear(y, x, n, x0)` of `src/quadriga_lib_helper_functions.hpp` (constant beyond
both ends) serves the ACF tables, the zenith table of `path_loss` and the 38.811 tables.

File-local (anonymous namespace or lambda in the one file that uses them): the symmetric square root
(`lsp_generate.cpp`), the probit (`los_probability.cpp`), the path-loss models and the random hE
(`path_loss.cpp`), the tables of section 11.

Phase-set machinery. `lsf_init_phases` and `lsf_draw` share one source file, `src/qd_gbsm_phases.cpp` (name: the
`qd_` prefix marks a file with several public functions, as `qd_scenpar.cpp`; "gbsm_phases" because the file holds
all phase sets of the model, not only the LSF set). It starts with the LSF generator index (table of 2.2), the
slice convention of the phase cube (slice 0 Rx side, slice 1 Tx side = mirror of slice 0) and the units, followed
by an anonymous namespace with the machinery, then the two documentation blocks and definitions. The machinery is
written for any phase set, not only the LSF set (explicit parameters, no captured state, no LSF-specific
constants except in `qd_gbsm_lsf_generators`); the later SSF port adds its own generator tables and the functions
that create and evaluate its phase sets to this file, so the machinery never moves to a header (if the SSF
functions had to live elsewhere, the functions below would move unchanged, as `static inline`, into the internal
header):

```cpp
namespace
{
    // One generator of a phase set
    struct qd_gbsm_gen { double lambda; bool uniform; arma::uword mirror; bool bias; };

    // Number of sinusoids of a preset; throws "Input 'preset' is not a known preset." (the one preset lookup)
    arma::uword qd_gbsm_n_coeff(const std::string &preset);

    // The 11 LSF generators (index table) from a completed scenpar
    std::vector<qd_gbsm_gen> qd_gbsm_lsf_generators(const std::unordered_map<std::string, double> &sp);

    // Frequency tables, one per distinct decorrelation distance of 'gen' (in order of first appearance):
    // sos_preset(preset, freq[t], &acf[t], &dist[t], lambda_t); tab[g] = table index of generator g; 'preset' is
    // known (checked by the caller with qd_gbsm_n_coeff)
    void qd_gbsm_freq(const std::string &preset, const std::vector<qd_gbsm_gen> &gen, std::vector<arma::mat> &freq,
                      std::vector<arma::vec> &acf, std::vector<arma::vec> &dist, std::vector<arma::uword> &tab);

    // Positions for the evaluation: R = rx_pos, T = tx_pos broadcast to n_link rows; z = 0 in 2D mode
    void qd_gbsm_positions(const arma::mat &rx_pos, const arma::mat &tx_pos, bool horizontal, arma::mat &R, arma::mat &T);

    // Phase offsets of floor q, [n_coeff]: 2·pi·(q mod N)·n/N + psi_(q div N)(n)
    arma::vec qd_gbsm_floor_offset(arma::uword n_coeff, arma::uword q);

    // Normalized values of a phase set at a set of links
    // - gen, freq, acf, dist, tab: generator table and frequency tables (qd_gbsm_lsf_generators, qd_gbsm_freq)
    // - phase [n_coeff, n_gen, 2]: slice 0 Rx side, slice 1 Tx side (mirror of slice 0)
    // - R, T [n_link, 3], n_link >= 1: positions from qd_gbsm_positions (T unused without dual mobility)
    // - floor_id: [n_link] or nullptr (all on floor 0); the offsets go to both slices (both devices on the
    //   link's floor)
    // - val [n_link, n_gen]: column g Normal(0,1), or Uniform(0,1) (map 0.5·erfc(-x/sqrt(2))) if gen[g].uniform
    void qd_gbsm_eval(const std::vector<qd_gbsm_gen> &gen, const std::vector<arma::mat> &freq,
                      const std::vector<arma::vec> &acf, const std::vector<arma::vec> &dist,
                      const std::vector<arma::uword> &tab, const arma::cube &phase,
                      const arma::mat &R, const arma::mat &T, const arma::uvec *floor_id,
                      bool dual_mobility, arma::mat &val);

    // Bias removal of the generators with gen[g].bias (Normal fields), in place
    // - P0 [n_coeff, n_gen]: Rx-side phases; U [n_pos, 3], n_pos > 2: all positions where the fields are evaluated
    void qd_gbsm_remove_bias(const std::vector<qd_gbsm_gen> &gen, const std::vector<arma::mat> &freq,
                             const std::vector<arma::uword> &tab, arma::mat &P0, const arma::mat &U);
}
```

`qd_gbsm_eval` implements steps 3 to 6 of 3.2 for `n_gen` generators (`lsf_draw` copies `val` into `rnd`);
`qd_gbsm_remove_bias` implements step 3 of 3.1 (the caller builds `U`); `qd_gbsm_floor_offset` implements 3.2.1.
These functions are reused within the file or written for the SSF phase sets, so they are functions of the
anonymous namespace, not lambdas. They group the `sos_val` / `sos_remove_bias` calls by frequency table (one call
per table and side with all phase columns that use it), so that a set with hundreds of generators at one
decorrelation distance (the SSF sets) needs one call per table. A generator `g` with `mirror != g` is a pair
member; pairs need no extra step (3.2 step 5).

### 3.1 `lsf_init_phases`

Source: `init_sos.m` lines 112 to 160 (LSF part), `qd_sos.m` `init`, `remove_bias`.

**Validation** (after 3.0.2 for the inputs that are given; `rx_pos` and `tx_pos` are optional here):

1. `seed >= -1`, else `Input 'seed' must be >= -1.` (the message of `sos_init_phases`, checked here so that it
   comes before any other work).
2. `preset` known (`qd_gbsm_n_coeff`, which uses `qd_sos_preset_lookup`), else `Input 'preset' is not a known
   preset.` (the message of `sos_rand`); `n_coeff` from the lookup (the only lookup of the call).
3. scenpar completed and validated (3.0.3); this covers the 10 decorrelation-distance keys of 2.2.
4. `rx_pos` (if given): 3 columns, finite. `tx_pos` (if given): 3 columns, finite; if `rx_pos` is given as well,
   1 or `n_link` rows.
5. `dual_mobility && remove_bias && rx_pos != nullptr && n_link > 2 && tx_pos == nullptr`:
   `Input 'tx_pos' is required for the bias removal with dual mobility.`

**Algorithm**

1. `sp` = completed scenpar; generator table of 2.2 with `lambda'_g = lambda_g > 0 ? lambda_g : 0.1`.
2. `P0 = sos_init_phases(n_coeff, 11, seed)`, `[n_coeff, 11]` (slice 0, Rx side; documented contract).
3. Bias removal, only if `remove_bias && rx_pos != nullptr && n_link > 2` (`n_link = rx_pos->n_rows`; with fewer
   positions, including 0, no SOS function is called):
   - Positions `R`, `T` from `qd_gbsm_positions` (Tx broadcast to `n_link` rows, z = 0 in 2D mode).
   - Positions of the field: `U = R` in single mobility (`tx_pos` is not used), `U = [R; T]` (`2·n_link` rows,
     Rx rows first) in dual mobility.
   - For every Normal generator g = 0..8: `sos_remove_bias(F_g, P0.col(g), U)` (one device; `F_g` the table
     `tab[g]` of `qd_gbsm_freq`). `P0.col(g)` stands for a `[n_coeff, 1]` matrix copied out and written back.
     Generators that share a table are corrected in one call (same result: the offset is computed per column).
   - Generators 9 and 10 are never corrected.
   - This step is `qd_gbsm_remove_bias(gen, freq, tab, P0, U)` (3.0.5).
4. Mirror: `P1(:, g) = P0(:, m(g))` for all g (slice 1, Tx side): the Tx side is the mirror of the corrected Rx
   side, so the mirror relation holds exactly with and without bias removal.
5. `phase` = `[n_coeff, 11, 2]`, slice 0 = `P0`, slice 1 = `P1`.

Why one correction over the union in dual mobility: the field of a generator is shared by both devices
(the Tx side of ASD is the ASA field), so its sample mean is removed over all positions where it is evaluated.
Mirror relations and the swap reciprocity of `lsf_draw` then hold exactly, and single and dual mobility share one
code path. Swapping Tx and Rx positions gives the same union and therefore the same phases, equal to rounding
(the sums over `[R; T]` and `[T; R]` are formed in another order).

**Properties to document**: same seed gives the same phases; without bias removal slice 0 equals
`sos_init_phases(n_coeff, 11, seed)`; slice 1 is always the mirror of slice 0; the sample mean of every Normal field
over the positions is 0 (single mobility: over `rx_pos`; dual mobility: over the Rx and Tx positions together);
consequently the link mean of the `lsf_draw` values is 0 in single mobility, and in dual mobility for DS, KF, SF,
XPR and TOA when all links have the same Tx-Rx distance (the normalization `1/sqrt(2 + 2·A(D))` is then the same
for all links), while the ASD/ASA and ESD/ESA columns have a zero link mean only approximately (their two devices
see different fields, each corrected over the union of the positions); bias removal over several floors: all positions in `rx_pos`
are treated as one field (floor 0), because one phase correction per generator cannot zero the means of several
floor fields at once; for a drop with several floors pass the positions of one floor (e.g.
the links of floor 0) or set `remove_bias = false`, the fields of other floors keep their sample mean; `tx_pos` is
ignored without dual mobility; uncorrected generators 9, 10.

Difference to QuaDRiGa (stated in section 10 and the release notes; not in the documentation block, which describes
only the library's rule): `init_sos.m` removes the bias before the mirroring, and the mirroring overwrites the
Rx-side phases of ASD with the Tx-side phases of ASA (and vice versa; the same for ESD/ESA). The correction these
phases carry was computed with the partner's frequencies, so it holds only when the two decorrelation distances
are equal. This affects single mobility as well: whenever ASD/ASA or ESD/ESA have different decorrelation
distances (e.g. UMa LOS: ASD 18 m, ASA 15 m), MATLAB's ASD/ASA and ESD/ESA fields keep a non-zero drop mean, while
the library removes it; the bias-removed values change against today in single mobility too, and in dual mobility
in general.

**Output**: `phase` `[n_coeff, 11, 2]`, phases in rad: uncorrected generators in [-pi, pi) (`sos_init_phases`),
corrected generators in [-pi, pi) as returned by `sos_remove_bias`.

### 3.2 `lsf_draw`

Source: `gen_lsf_parameters.m` lines 192 to 226 (`val` of the 8 generators; its dual-mobility coupling is not
taken, step 5), `qd_sos.m` `val`/`eval_mode`/`sos_eval`.

**Validation**: 3.0.2 for `tx_pos`, `rx_pos`; `preset` known (`Input 'preset' is not a known preset.`);
scenpar completed and validated (3.0.3); `phase` must be `[n_coeff, 11, 2]` with `n_coeff` of the preset:
`Input 'phase' must have size [<n_coeff>, 11, 2] for preset '<preset>'.`; `phase` finite:
`Input 'phase' must be finite.`; `floor_id` (if given) `n_link` elements.

**Algorithm**

0. `n_link = 0`: `rnd` is resized to `[0, 11]` and the function returns after the validation; no SOS function is
   called (they throw on empty positions).
1. Positions `R`, `T` from `qd_gbsm_positions` (Tx broadcast to `n_link` rows; z = 0 in 2D mode). In
   single mobility `T` is not used (the values do not depend on `tx_pos`).
2. Generator table of 2.2 and the frequency tables from `qd_gbsm_freq` (`lambda'_g`); `F_g, acf_g, dist_g` denote
   the table `tab[g]` of generator g; `uniform_g` = true for g = 9, 10.
3. Floors: `fl(l) = floor_id ? (*floor_id)(l) : 0`. Links are grouped by distinct floor ID. For floor `q` the phases
   are `P0_q = P0 + Delta_q`, `P1_q = P1 + Delta_q` (the same offset vector added to every column of both slices),
   `Delta_q` from 3.2.1. Floor 0 has `Delta_0 = 0`, so `floor_id` all 0 gives exactly the values of
   `floor_id = nullptr`. In dual mobility `floor_id` is one value per link and its offsets go to both sides: both
   devices of a link are on the link's floor. A Tx device on another floor than its Rx (e.g. a D2D link between
   two floors) is evaluated on the floor of the Rx, so the same Tx device sees different fields on links to Rx
   devices on different floors. The documentation blocks (C++, MEX, Python) state this; a separate Tx floor ID is
   not part of the API.
4. Fields (all generators evaluated as Normal, `uniform = false`), for every floor group (links `L`) and generator
   g, with `sos_val<double>` in one-device mode and `use_kernel = 0`:
   - Rx side: `a_g = sos_val(F_g, P0_q.col(g), R.rows(L))`.
   - Tx side (dual mobility only): `b_g = sos_val(F_m(g), P1_q.col(g), T.rows(L))`, i.e. the field of the mirror
     generator `m(g)` at the Tx position (`P1(:, g) = P0(:, m(g))`; `F_m(g)` equals `F_g` scaled by
     `lambda'_g / lambda'_m(g)`). For the self-mirrored generators this is `g`'s own field at the Tx position.
   The calls are grouped by frequency table and side: one `sos_val` call per table evaluates all phase columns
   that use it (same values as one call per generator).
5. Combination, per link `l`, with `D = norm(R(l) - T(l))` (projected positions in 2D mode) and
   `A_g(D) = qd_interp_linear(acf_g, dist_g, D)` (linear, constant beyond the last sample point):
   - single mobility: `x_g = a_g`;
   - dual mobility, `m(g) = g`: `x_g = (a_g + b_g) / sqrt(2 + 2·A_g(D))` (both devices on one map; unit variance);
   - dual mobility, pairs (3, 4) and (5, 6): `x_g = (a_g + b_g) / sqrt(2)` (two independent fields, unit
     variance). No further step: the pair `x_g = (f_g(R) + f_m(T))/sqrt(2)`, `x_m = (f_m(R) + f_g(T))/sqrt(2)`
     already has the correlation `Corr(x_g, x_m) = (A_g(D) + A_m(D))/2`, which is 1 at `D = 0` (then `x_g = x_m`
     exactly) and falls with the ACF of the two generators at the Tx-Rx distance. QuaDRiGa multiplies the pair once
     more by `sqrtm([[1, A_g(D)], [A_m(D), 1]])`; for equal decorrelation distances (`A_g = A_m = c`) this gives
     the covariance `[[1 + c², 2c], [2c, 1 + c²]]` (variance up to 2 at small distances, correlation
     `2c/(1 + c²)`); for unequal ones the variances exceed 1 as well (e.g. 1.73 and 1.29 for `A_g = 0.9`,
     `A_m = 0.5`). The library does not, so that columns 3 to 6 stay
     Normal(0,1).
6. `rnd` `[n_link, 11]`: column g = `x_g` for g = 0..8; `0.5 · erfc(-x_g / sqrt(2))` for g = 9, 10.

Steps 3 to 6 are `qd_gbsm_eval` (3.0.5) with the 11 LSF generators; the offsets of step 3 come from
`qd_gbsm_floor_offset`.

This is equivalent to the device modes of `sos_val` (two devices on one map for the self-mirrored generators, two
maps with `freq_scale_b = lambda'_g / lambda'_m(g)` for the pairs) but evaluates each field with its own frequency
table, so that swapping `tx_pos` and `rx_pos` repeats exactly the same `sos_val` calls in swapped roles: the swap
reciprocity holds bit for bit, also for unequal decorrelation distances and with bias-removed phases.

**Properties to document** (and test): columns 0 to 8 Normal(0,1), 9 and 10 Uniform(0,1); the value of a link does
not depend on the other links of the call; 2D mode: the values do not depend on z; dual mobility: swapping
`tx_pos` and `rx_pos` leaves columns 0, 1, 2, 7, 8, 9, 10 unchanged and swaps 3 with 4 and 5 with 6, also for
unequal decorrelation distances and with bias removal (when both calls use the same phase cube); dual mobility:
columns 3 to 6 are Normal(0,1) with `Corr(x_3, x_4) = (A_3(D) + A_4(D))/2` and `Corr(x_5, x_6) = (A_5(D) +
A_6(D))/2` at the Tx-Rx distance `D`, equal at `D = 0`; floors: fields of different floor IDs are
uncorrelated at equal horizontal positions (3.2.1), and the floor ID is the caller's floor index of the UT (for
the TR UT heights `hUT = 3(n_fl - 1) + 1.5` of Table 7.2-1, taken from TR 36.873: `floor_id = n_fl - 1`; outdoor
UTs 0); precision: the sinusoids are evaluated with the AVX2 kernel where available (single-precision sin/cos,
deviation below `4e-7 · sqrt(2 · n_coeff)`); `preset` and `horizontal` must be those of `lsf_init_phases` (a preset with the same `n_coeff`, e.g. `Exp256` for `Comb256`,
cannot be detected).

#### 3.2.1 Floor phase offsets

```
N = n_coeff;  q = floor ID (arma::uword);  k = q mod N;  b = q div N
Delta_q(n) = 2·pi · k · n / N + psi_b(n),        n = 0 .. N-1
psi_0(n)   = 0
psi_b(n)   = 2·pi · u_n                         b >= 1; rng = qd_sos_rng(b), u_n = qd_sos_uniform(rng), n = 0 .. N-1 in order
```

`qd_sos_rng(b)` seeds `std::mt19937_64` with `b`, and `qd_sos_uniform` maps one raw 64-bit output to
`(x >> 11) · 2^-53`; both are portable across standard libraries (unlike
`std::uniform_real_distribution`), so a test can retype the rule with `std::mt19937_64 gen(b)`.

Properties: for two floors in the same block of N IDs the cross-correlation of their fields at zero horizontal lag
(the same x, y) is `(1/N) · sum_n cos(Delta_q(n) - Delta_q'(n)) = 0` exactly (k differs, the sum over a full period
of a discrete cosine vanishes); for floors in different blocks it is a random number with standard deviation
`1/sqrt(2N)` (0.044 for 256 sinusoids). At non-zero horizontal lag the cross-covariance of two floors is not zero
(up to 0.11 to 0.16 for the 256-sinusoid presets, 0.07 to 0.10 for 1024, at lags of 0 to 60 m in 2D mode), the
same level as for pseudo-random offsets; the documentation block states "uncorrelated at equal horizontal
positions", not "independent". The same offsets serve both slices and all generators, so mirror relations and
reciprocity are kept per floor. The offsets are `qd_gbsm_floor_offset` of 3.0.5 (anonymous namespace in
`src/qd_gbsm_phases.cpp`).

### 3.3 `path_loss`

Source: `get_pl.m` (complete), TR Table 7.4.1-1 and note 1, clause 7.4.3.1.

#### 3.3.1 Validation (after 3.0.2)

1. `d2d_in` (if given): `n_link` elements; finite and >= 0: `Input 'd2d_in' must be finite and >= 0.`
2. `rnd_he` (if given): `n_link` elements; values in [0, 1]: `Input 'rnd_he' must be in the range from 0 to 1.`
3. `pl_model`: empty, or one of the 10 names in 3.3.4: `Path-loss model '<name>' is not supported.`
4. Required parameters of the model (3.3.4) present: `Path-loss model '<model>' requires the parameter '<key>'.`
   Every `plpar` entry that the model reads (required or optional) must have exactly one value:
   `Path-loss parameter '<key>' must have one value.` (no conf file in `data/config` or in QuaDRiGa-NG has a
   `PL_` key with several values). Entries not read by the model are ignored.
5. Breakpoint factors: `E` (dual_slope; nlos with LOS term), `E1`, `E2` (tripple_slope) must be > 0:
   `Path-loss parameter '<key>' must be > 0.`; `winner_los`: `A1 != A2`, else
   `Path-loss parameters 'A1' and 'A2' must differ for model 'winner_los'.`
6. scenpar completed and validated (3.0.3; it holds `LSP_fc_min` and the SF keys of the default `sf_sigma`).

#### 3.3.2 Distances

Per link (frequency-independent):

```
d3 = d3D,  d2 = d2D                                  geometric distances (TR: d3D-out + d3D-in, eq. 7.4-1)
if d2d_in given and d2d_in(l) > 0:                    legacy outdoor distance (gen_o2i_loss.m lines 185 to 223,
                                                      get_pl.m lines 177 to 186)
    din  = min(d2d_in(l), d2)
    d3in = d2 > 0 ? d3 · din / d2 : d3                indoor part of the 3D distance (MATLAB: NaN -> Inf -> d3)
    d3in = min(d3in, d3)
    if d3 - d3in < 10:  d3in = d3 - 10                minimum outdoor distance 10 m
    d3in = max(d3in, 0)
    p    = d3 > 0 ? d3in / d3 : 0                     d3D = 0 gives an indoor fraction of 0
    d3   = d3 - d3in
    d2   = d2 · (1 - p)
d3 = max(d3, 0.1);  d2 = max(d2, 0.1)                 get_pl.m lines 189, 190
```

Without `d2d_in` (TR, default) the geometric distance between the Tx and the indoor Rx is used; it is already
`d3D-out + d3D-in`. Passing `d2d_in` (the indoor 2D distance of each link's UT from `o2i_loss`; 0 for outdoor UTs,
whose distance stays geometric) selects QuaDRiGa's legacy mode, in which the path-loss formulas see only the
outdoor part. The O2I loss itself is never added by `path_loss`; the caller adds the output of `o2i_loss`.

`f` is not clamped by `LSP_fc_min` in the path loss.

#### 3.3.3 Default shadow-fading standard deviation

`sf_sigma(l, k) = sigma_2(l, k)` of 3.0.4 (SF index, raw geometry, not the O2I-scaled distances), not `SF_sigma + SF_delta · lg(f)`, so that passing `sf_sigma` to `lsp_generate` is neutral for models
without `sig` keys. Models with `sig`-type keys replace it as stated below. For all v19 files without `sig` keys this
is `SF_sigma` (they have `SF_delta = SF_omega = SF_kappa = SF_tau = SF_beta = 0`). The default is computed per
link and frequency with `qd_gbsm_lsp_stats` restricted to SF (no 8-LSP intermediate); with `sf_sigma` requested,
an active SF frequency term with `SF_omega + f' <= 0` throws (3.0.4) after the scenpar validation.

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
| `dual_slope` | A1 or A, B, E | A2 = A1, C = 0, D = 0, hE = 0, hE_rand = 0, sig1/sig2 optional |
| `tripple_slope` | A1 or A, B, E1 or E | A2 = A1, A3 = A2, C = 0, D = 0, E2 = E1, hE1 = 0, hE2 = 0, sig1..sig3 optional |
| `nlos` | An, Bn; plus the `dual_slope` keys if LOSmax != 0 | Cn = Dn = D1n = D2n = D3n = En = E1n = E2n = E3n = Fn = G1n = 0, G2n = 1, LOSmax = 1, Ax = Bx = Cx = 0 (term inactive unless one of them is present), sig optional |
| `satellite` | none | A = 20, B = 32.45, C = 20, D = 0, usePLa = 1 |

Formulas per link `l` and frequency `k` (`get_pl.m` line numbers in brackets):

- **constant** [210]: `pl = [A]`.
- **logdist** [201]: `pl = [A]·lg(d3) + [B] + [C]·lg(f)`; `sf = [SF]` if present.
- **logdist_simple** [207]: `pl = [A]·lg(d3) + [B]`.
- **winner_los** [213 to 245]: `hb = max(hBS, 1.5)`, `hm = max(hUT, 1.5)` of the link itself;
  `G = B1 + C1·lg f + D1·lg hb + E1·lg hm`, `H = B2 + C2·lg f + D2·lg hb + E2·lg hm`,
  `bp = 10^((H - G)/(A1 - A2))`;
  `d2 <= bp`: `pl = A1·lg d2 + B1 + C1·lg f + D1·lg hb + E1·lg hm + F1·hm`, `sf = sig1`;
  else `pl = A2·lg d2 + B2 + C2·lg f + D2·lg hb + E2·lg hm + F2·hm`, `sf = sig2`. (`get_pl.m` takes the
  breakpoint from the mean heights of the call, i.e. of one track, and uses `C1` in the second branch, line 241;
  both differ from the library only where the heights vary within one MATLAB call, or beyond the breakpoint when
  `C1 != C2`. With `C2` the loss is continuous at `bp`: `A2·lg bp + H + F2·hm = A1·lg bp + G + F1·hm` holds when
  `F1 = F2`, as in all WINNER files.)
- **winner_nlos** [247 to 278]: `hb = max(hBS, 1.5)`, `hm = max(hUT, 1.5)`;
  `L_j = (A_j + Ah_j·lg hb)·lg d2 + B_j + C_j·lg f + D_j·lg hb + E_j·lg hm + F_j·hm` (j = 1, 2, 3);
  `pl = L_1` for `f < 1.5`, `L_2` for `1.5 <= f < 2`, `L_3` for `f >= 2`; `sf` default.
- **winner_pathloss** [280 to 289]: `pl = [A]·lg d3 + [B] + [C]·lg(f/5) + [X]`; `sf = [SF]` if present.
- **dual_slope** [291 to 323]: `hE` = `[hE|0]`, or per link from 3.3.6 if `[hE_rand|0] != 0` and `rnd_he` is given;
  `hb = hBS; if hb < 1.1·hE: hb = 1.1·hE; if hb < 0.1: hb = 0.1`; the same for `hm` from `hUT` with the fixed hE,
  but `hm = max(hUT, 0.1)` without the 1.1·hE clamp when hE is the random one of 3.3.6. Reason: the TR set of the
  random hE ends at `hUT - 1.5`, so `hUT - hE >= 1.5` always holds, and the clamp would act at the TR UT heights of
  floors 7 and 8 (hUT 19.5 m with hE 18 m, hUT 22.5 m with hE 21 m: `h'UT` 1.8 or 2.1 m instead of 1.5 m, the loss
  beyond the breakpoint 1.4 or 2.6 dB too low). The clamp of `hb` stays: it never acts at the TR BS height of UMa
  (25 m > 1.1 · 21 m) and keeps `hb - hE` positive for a lower BS;
  `dBP = E·f·(hb - hE)·(hm - hE)`; `dBP3 = sqrt(dBP² + (hb - hm)²)`;
  `d2 <= dBP`: `pl = PL1(d3)`, `sf = [sig1]` if present; `d2 > dBP`: `pl = PL1(dBP3) + A2·lg(d3/dBP3)` with the
  3D breakpoint distance, `sf = [sig2]` if present. This is Table 7.4.1-1 for UMa, UMi, InH and InF. For RMa and
  SMa LOS the TR uses the 2D breakpoint distance, `PL1(dBP) + 40·lg(d3D/dBP)`; the difference
  `(A1 - A2)·lg(dBP3/dBP) + D·(dBP3 - dBP)` is at most 0.034 dB over the TR ranges (RMa: hBS 10 to 150 m, hUT 1 to
  10 m, 0.5 to 30 GHz, maximum 0.0341 dB at 0.5 GHz and hUT 1 m; SMa: hBS 25 to 35 m, hUT 1 to 14 m, 0.5 to 37 GHz,
  maximum 0.0313 dB), and the library keeps the 3D form for all scenarios (1.2).
- **tripple_slope** [325 to 378]: `hb = max(hBS, 1.1·hE1, 1.1·hE2, 0.1)` (same for `hm`);
  `dBP1 = E1·f·(hb - hE1)·(hm - hE1)`, `dBP2 = E2·f·(hb - hE2)·(hm - hE2)`, `dBP1 = min(dBP1, dBP2)`;
  `dBPj3 = sqrt(dBPj² + (hb - hm)²)`; `LBP1 = PL1(dBP13)`, `LBP2 = LBP1 + A2·lg(dBP23/dBP13)`;
  `d2 <= dBP1`: `PL1(d3)`, `sf = sig1`; `dBP1 < d2 <= dBP2`: `LBP1 + A2·lg(d3/dBP13)`, `sf = sig2`;
  `d2 > dBP2`: `LBP2 + A3·lg(d3/dBP23)`, `sf = sig3` (each `sf` only if the key is present).
- **nlos** [380 to 434] plus the keys `PL_LOSmax`, `PL_Ax`, `PL_Bx`, `PL_Cx` (4.3): `hb = max(hBS, 0.1)`,
  `hm = max(hUT, 0.1)`;
  ```
  PLn = An·lg d3 + Bn + Cn·lg f + Dn·lg hb + D1n·lg(hb)/hb + D2n·lg(hb)/hb² + D3n·hb
      + En·lg hm + E1n·lg(hm)/hm + E2n·lg(hm)/hm² + E3n·hm + Fn·lg(hb)·lg(d3) + G1n·(lg(G2n·hm))²
  pl  = PLn
  if [LOSmax|1] != 0:      pl = max(pl, PL_dual_slope)     (dual_slope above with the same plpar incl. hE_rand, its sf unused)
  if Ax, Bx or Cx present: pl = max(pl, [Ax|0]·lg d3 + [Bx|0] + [Cx|0]·lg f)
  sf = [sig] if present
  ```
- **satellite** [436 to 476]: `sa = min(hBS / d3, 1)` (absolute Tx height, as MATLAB); if `sa <= 0`: `pl = +Inf`; else
  `alpha = asin(sa)` [rad], `PLa = [usePLa] != 0 ? Z(f) / sa : 0` with `Z(f)` the zenith attenuation of Appendix 11.1,
  linearly interpolated over f in GHz with constant values beyond 1 and 100 GHz (`qd_interp_linear(Z, f_grid, 93,
  f)`: linear, constant beyond both ends, strictly increasing grid); `pl = A·lg d3 + B + C·lg f + D·lg(alpha) + PLa`.

`sf_sigma` is the default (3.3.3) wherever a model does not set it.

#### 3.3.5 Edge rules

- `n_link = 0`: empty outputs.
- `fc` may be any positive frequency; the models are evaluated as written, also outside their validity range.
- Missing `PL_hE_rand`, `PL_LOSmax`, `PL_Ax/Bx/Cx` keep QuaDRiGa's behaviour (fixed hE, maximum with the LOS term,
  no extra term); `PL_hE_rand = 1` without `rnd_he` uses the fixed `PL_hE`.

#### 3.3.6 UMa random environment height (TR Table 7.4.1-1 note 1)

Per link, frequency-independent, in `dual_slope` (and the LOS term of `nlos`) when `[hE_rand|0] != 0` and `rnd_he`
is given; `u = rnd_he(l)`, `hUT` raw, `d` = the `d2` of 3.3.2:

```
g  = (d <= 18) ? 0 : 1.25 · (d/100)³ · exp(-d/150)
C' = (hUT < 13) ? 0 : (hUT <= 23 ? ((hUT - 13)/10)^1.5 : 1)
C  = C' · g;   p1 = 1 / (1 + C)
n  = (hUT - 1.5 >= 12) ? floor((hUT - 13.5)/3 + 1e-9) + 1 : 0          size of the set {12, 15, ..., hUT - 1.5}
if C == 0 or n == 0 or u < p1 or p1 >= 1:  hE = 1
else:  j = min(floor((u - p1) / (1 - p1) · n), n - 1);  hE = 12 + 3·j
```

`p1 >= 1` occurs where `C` is below the rounding of `1 + C` (about 1.1e-16, e.g. beyond about 7.5 km at
`hUT = 20 m`); the other heights then have a probability below 1e-16, and without the rule `u = 1` would give
`0/0` and a NaN path loss.

The fixed `[hE]` is not used for such a link, and `hm` gets no 1.1·hE clamp (3.3.4). The value 1 m is the TR value
(the v19 UMa files also have `PL_hE = 1`).

`rnd_he` is column 10 of `lsf_draw`, a spatially consistent Uniform(0,1) value with the decorrelation distance
`SF_lambda`. Along a track hE therefore switches between its discrete values wherever `u` crosses a threshold
(`p1` itself varies with `d2D`), and beyond the breakpoint the path loss steps there (e.g. 3.7 dB between hE 1 m
and 18 m at 1 GHz, d2D 300 m, hUT 20 m, hBS 25 m). This is the TR model (hE is a per-link draw); a caller that
wants no steps along a track segment evaluates `rnd_he` once per segment (10.2).

#### 3.3.7 Documentation block (must state)

The 10 models with their formulas or a pointer to the parameter names; the distance clamps; the O2I distance rule
(geometric distance without `d2d_in`; with it the outdoor distance, computed with the indoor 3D part and the 10 m
minimum outdoor distance; an entry `d2d_in = 0` marks an outdoor link, which keeps the geometric distance and gets
no 10 m rule) and that the O2I loss is not added; that `dual_slope` uses the 3D breakpoint distance
beyond the breakpoint, which differs from Table 7.4.1-1 for RMa and SMa LOS by at most 0.034 dB; the default
`sf_sigma`; the random hE rule, without the 1.1·hE clamp of the UT height, and that the spatially consistent
`rnd_he` makes hE and the path loss beyond the breakpoint step along a track (evaluate `rnd_he` once per segment to
avoid this); the new keys `PL_hE_rand`, `PL_LOSmax`, `PL_Ax`, `PL_Bx`, `PL_Cx`; that
`f` is not clamped; required keys, the one-value rule and the error for a missing key; `[n_link, n_freq]` sizes.
`winner_los` is documented with `C2` beyond the breakpoint and per-link breakpoint heights (no reference to the
MATLAB behaviour in the block).

### 3.4 `lsp_generate`

Source: `gen_lsf_parameters.m` (lines 228 to 268 and 309 to 361), `qd_builder.m` `lsp_vals`; TR 7.5
step 4, Tables 7.5-6 to 7.5-12, 7.6.5, 7.6.9.

#### 3.4.1 Validation (after 3.0.2)

1. `rnd`: `[n_link, 11]`: `Input 'rnd' must have size [n_link, 11].`; finite: `Input 'rnd' must be finite.`
2. `sf_sigma` (if given): `[n_link, n_freq]`: `Input 'sf_sigma' must have size [n_link, n_freq].`; finite and
   >= 0: `Input 'sf_sigma' must be finite and >= 0.` (a negative value would flip the sign of the SF).
3. scenpar completed and validated with `lsp_xcorr` and the return value (3.0.3); this covers `LSP_fc_min`,
   `DS_VS`, `DS_VS_a`, `DS_VS_b`, `absTOA_Lmax` and the cross-key rule of 4.2 (`DS_VS > 0` with
   `DS_VS_a = DS_VS_b = 0`).
4. Not positive definite (return value of `scenpar_validate` false):
   `LSP cross-correlation matrix in 'scenpar' is not positive definite.` (both roots; MATLAB refuses as well,
   `gen_lsf_parameters.m` lines 120 to 123). The same message is thrown when `qd_gbsm_chol8` fails on the permuted
   matrix of 3.4.2 step 1 although `scenpar_validate` accepted `R` (possible for a nearly singular `R`, whose pivots
   round differently in another order). This check runs before any output is touched.
5. Only if `gr_eps_re` or `gr_eps_im` is requested: column 9 of `rnd` in [0, 1]:
   `Column 9 of 'rnd' must be in the range from 0 to 1.`

#### 3.4.2 Algorithm

1. Root `Q` (8x8, our order DS, KF, SF, ASD, ASA, ESD, ESA, XPR) of `R = lsp_xcorr`:
   - `cholesky = true` (TR step 4, order `[SF, K, DS, ASD, ASA, ZSD, ZSA]`, XPR last): permutation
     `pi = [2, 1, 0, 3, 4, 5, 6, 7]` (TR position a holds our LSP `pi[a]`); `Rp(a, b) = R(pi[a], pi[b])`;
     `qd_gbsm_chol8(Rp, L)` (lower, `Rp = L·L^T`); `Q(pi[a], pi[b]) = L(a, b)`. Then `x = Q·s` equals the TR
     rule `x_TR = L·s_TR` with `s_TR[a] = s[pi[a]]`. In particular `x_SF = s_SF` (the SF field is passed through).
   - `cholesky = false` (QuaDRiGa's rule, MATLAB `sqrtm`): `Q` = symmetric square root of `R` (3.4.3, file-local).
2. Per link: `s = rnd(l, 0..7)^T`, `x = Q·s` (same `x` for all frequencies, TR clause 7.6.5).
3. Per link and frequency: `mu_i`, `sigma_i` from 3.0.4; if `sf_sigma` is given,
   `sigma_2 = sf_sigma(l, k)`. `v_i = mu_i + sigma_i · x_i`.
   `lsp(l, k, i) = 10^(v_i)` for i in {0, 3, 4, 5, 6}; `10^(0.1·v_i)` for i in {1, 2, 7}.
4. `clip_spreads = true` (TR clause 7.5 step 4): `lsp(:, :, 3) = min(., 104)`, `lsp(:, :, 4) = min(., 104)`,
   `lsp(:, :, 5) = min(., 52)`, `lsp(:, :, 6) = min(., 52)`. `lsp_mu`, `lsp_sigma` are not affected.
5. `lsp_mu(l, k, i) = mu_i`, `lsp_sigma(l, k, i) = sigma_i` (with the `sf_sigma` replacement).
6. `abs_toa(l)` (TR clause 7.6.9 with QuaDRiGa's activation rule): if `absTOA_mu > -30`:
   `min(10^(absTOA_mu + absTOA_sigma · rnd(l, 8)), absTOA_Lmax > 0 ? 2·absTOA_Lmax/c : +Inf)`; else 0.
7. Ground permittivity, `gen_lsf_parameters.m` lines 319 to 344, `f` unclamped, `U = rnd(l, 9)`:
   ```
   dry = 3            + j·0.003·f^1.34
   med = 30.4·f^-0.47 + j·0.18·f^1.05
   wet = 31.3·f^-0.48 + j·0.63·f^0.77
   U <= 0.5:  w = 2·U;          eps = (1 - w)·dry + w·med
   U >  0.5:  w = 2·(U - 0.5);  eps = (1 - w)·med + w·wet
   gr_eps_re = Re(eps), gr_eps_im = Im(eps)     (imaginary part >= 0, QuaDRiGa convention)
   ```
   TR eq. (7.6-40) writes the complex permittivity as `eps_r - j·sigma/(2·pi·f·eps0)`, i.e. with the opposite sign
   of the imaginary part. A caller (the later SSF port) that uses the TR's reflection formulas takes the complex
   conjugate, `gr_eps_re - j·gr_eps_im`.
8. EoD offset (v19 keys `EoD_offset_*`), with `f'`, `d`, `h` of 3.0.4 and raw `hUT`:
   ```
   e = EoD_offset_e + EoD_offset_e_gamma · lg f'
   if EoD_offset_b > 0:
       e += 10^( (EoD_offset_a + EoD_offset_a_gamma·lg f') · lg(max(EoD_offset_b, d))
                 + EoD_offset_c + EoD_offset_c_gamma·lg f' + EoD_offset_c_hUT · hUT )
   if EoD_offset_hcl > 0 and hUT < EoD_offset_hcl:
       e += (atan((EoD_offset_hcl - h)/d) - atan((hUT - h)/d)) · 180/pi
   eod_offset_deg(l, k) = e                       degrees, positive = up (= -mu_offset,ZOD of the TR)
   ```
   RMa NLOS and both RMa O2I files: TR Table 7.5-9 gives `mu_offset,ZOD = atan((35 - 3.5)/d2D) - atan((35 -
   1.5)/d2D)` with fixed heights for every link. The files encode it with `EoD_offset_hcl = 3.5`, i.e. with the
   actual heights. This generalizes the TR row: it equals the TR at hBS 35 m, hUT 1.5 m and is 0 for UTs at or
   above the clutter height of 3.5 m (RMa allows hUT up to 10 m). The comment of the three files says so.

#### 3.4.3 Symmetric square root (`cholesky = false`)

QuaDRiGa takes the root with MATLAB's `sqrtm`. The library computes the principal square root `S` of the symmetric
positive definite 8x8 matrix `A = lsp_xcorr` (`S` symmetric positive definite, `S·S = A`) with the coupled
Newton-Schulz iteration: matrix products only, no BLAS/LAPACK (9.1), file-local in `src/lsp_generate.cpp` (single
user), tested through `lsp_generate` (5.4).

```
s = tr(A)                                    (= 8 for a correlation matrix)
Y = A / s;   Z = I                           (8x8)
for k = 1 .. 100:
    M = Z·Y
    e = ||I - M||_F                          Frobenius norm, written out (no arma::norm)
    T = (3·I - M) / 2
    Y = Y·T;   Z = T·Z
    if e <= 1e-10: stop                      the update computed with this e is kept
S = sqrt(s) · Y;   S = (S + S^T) / 2
```

- Scaling: for a positive definite `A` all eigenvalues of `A / tr(A)` lie in (0, 1) (n = 8 > 1), so
  `||I - A/s||_2 < 1`, which is the convergence condition of the iteration; `Y` converges to `(A/s)^(1/2)`, `Z` to
  `(A/s)^(-1/2)`.
- Convergence criterion: `e <= 1e-10`. In the first phase the small eigenvalues `m` of `Z·Y` grow by a factor of
  about 9/4 per iteration (`m <- m·(3 - m)²/4`) while `e` decreases only slowly (it stays near 1 for one small eigenvalue); once all eigenvalues are near 1, `e` falls
  quadratically (`e_next ≈ 0.75·e²`), so the last update brings `Y` to rounding level. Iterations needed
  for a correlation matrix with smallest eigenvalue 1e-2: about 15; 1e-6: about 25; 1e-10: about 37.
- Iteration limit 100. It is reached only for nearly singular matrices (smallest eigenvalue below about 1e-12),
  whose root cannot be computed more accurately in double; the coupled iteration is stable, so the remaining
  iterations do not degrade `S` (`max abs(S·S - A)` about 1e-8 for a smallest eigenvalue near 1e-16). No exception:
  matrices that are not positive definite were rejected before (3.4.1 item 4).
- For the well-conditioned matrices of the conf files the result equals MATLAB's `sqrtm` to rounding (about
  1e-14).
- Test tolerance of `R^(1/2) · R^(1/2) = R` (5.4): `max abs(X·X - R) <= 1e-12` and `max abs(X - X^T) <= 1e-12`
  for the test matrices (smallest eigenvalue down to about 1e-6, where the residual is about 2e-14).

The Cholesky is `qd_gbsm_chol8` (3.0.5).

#### 3.4.4 Documentation block (must state)

Units of every slice; the mean/std formula incl. `LSP_fc_min`, the ESD terms and the floor, the limit of the std
at 0, the V/S term
(a caller with another hall sets `DS_VS`); both roots and the TR order; clipping; `sf_sigma`; same random values for all frequencies; the abs TOA
rule and bound (the InF and InH NLOS files set `absTOA_Lmax` for the hall or office of their calibration
scenario; a caller with another hall sets the key); the ground permittivity model and its sign convention, with the note that TR eq. (7.6-40) uses the
opposite sign of the imaginary part (take the conjugate for the TR reflection formulas); the EoD offset formula and
sign; that `sf_sigma` must be >= 0; that
`rnd = zeros(n_link, 11)` gives the median LSPs and the `lsp_mu` / `lsp_sigma` outputs without randomness
(replacement of `lsp_vals`, section 10).

### 3.5 `los_probability`

Source: TR Table 7.4.2-1, clause 7.6.3.3 eq. (7.6-18), 7.8.4 Table 7.8-7; `set_scenario.m` (lines
338 to 715).

#### 3.5.1 Validation (in this order)

1. `model` one of the 18 names of the header: `Model '<name>' is not supported.`
2. `d2d_out` finite and >= 0: `Input 'd2d_out' must be finite and >= 0.`; `n_link = d2d_out.n_elem` (0 allowed).
3. `h_bs`, `h_ut`: 1 or `n_link` elements (`Number of elements in 'h_bs' must be 1 or match 'd2d_out'.`), finite
   (`Input 'h_bs' must be finite.`). A single element applies to all links.
4. `d_clutter` NaN or finite > 0 (`Input 'd_clutter' must be NaN or > 0.`); `r_clutter` NaN or in [0, 1)
   (`Input 'r_clutter' must be NaN or in the range from 0 to 1, 1 excluded.`); `h_clutter` NaN or finite >= 0
   (`Input 'h_clutter' must be NaN or >= 0.`). Checked for every model, used only by InF and SMa. A subnormal
   `d_clutter` passes and gives `kinv = +Inf`: the clutter attenuation `exp(-d·k)` is evaluated as 1 at `d = 0`
   and an SMa clutter type with `h_x <= hU` adds no term, so no `0·Inf` = NaN arises (1 at `d = 0`, 0 beyond for
   a blocked UT).
5. If `los_soft != nullptr`: `rnd_g` and `fc` are required (`Output 'los_soft' requires 'rnd_g' and 'fc'.`);
   `rnd_g` `n_link` elements, finite; `fc` non-empty, finite > 0. If `los_soft == nullptr`, `rnd_g` and `fc` are
   ignored (not validated).

#### 3.5.2 Models (`d = d2d_out(l)`, `hB = h_bs`, `hU = h_ut`; all results limited to [0, 1])

| Model | p_LOS |
|---|---|
| `38.901_RMa` | `d <= 10`: 1; else `exp(-(d - 10)/1000)` |
| `38.901_UMi` | `d <= 18`: 1; else `18/d + exp(-d/36)·(1 - 18/d)` |
| `38.901_UMa` | `d <= 18`: 1; else `(18/d + exp(-d/63)·(1 - 18/d)) · (1 + C'(hU)·5/4·(d/100)³·exp(-d/150))`, `C' = 0` for `hU <= 13`, `((hU - 13)/10)^1.5` for `13 < hU <= 23`, 1 for `hU > 23` |
| `38.901_InH_Mixed` | `d <= 1.2`: 1; `1.2 < d < 6.5`: `exp(-(d - 1.2)/4.7)`; `d >= 6.5`: `exp(-(d - 6.5)/32.6)·0.32` |
| `38.901_InH_Open` | `d <= 5`: 1; `5 < d <= 49`: `exp(-(d - 5)/70.8)`; `d > 49`: `exp(-(d - 49)/211.7)·0.54` |
| `38.901_InF_SL`, `_DL` | `exp(-d·kinv)`, `kinv = -ln(1 - r)/d_clutter` |
| `38.901_InF_SH`, `_DH` | see below |
| `38.901_InF_HH` | 1 |
| `38.901_SMa` | `d <= 10`: 1; else see below |
| `38.811_DenseUrban`, `_Urban`, `_Suburban`, `_Rural` | `0.01 ·` linear interpolation of the table of Appendix 11.2 at the elevation `atan2(hB, d)` in degrees (`qd_interp_linear` over the elevation grid) |
| `QuaDRiGa_Industrial` | `min(1, exp(-(d + 20)/80) · exp(min(hB, 8)/5 - 1))` |
| `LOS` / `NLOS` | 1 / 0 |

The limit to [0, 1] matters for `38.901_UMa`: for hU > 13 m the TR formula exceeds 1 slightly just above 18 m
(up to 18.29 m with a maximum of 1.0038 for hU 20 m, up to 18.47 m with 1.0060 for hU 22.5 m).

InF defaults (TR 38.901 clause 7.8.4, Table 7.8-7), used where the argument is NaN:

| Model | r | d_clutter [m] | h_c [m] |
|---|---|---|---|
| SL | 0.2 | 10 | (unused) |
| DL | 0.6 | 2 | (unused) |
| SH | 0.2 | 10 | 2 |
| DH | 0.6 | 2 | 6 |

InF-SH, InF-DH: `kinv0 = -ln(1 - r)/d_clutter`;
- `hU >= h_c`: p = 1 (UT above the clutter);
- else if `hB > hU`: `p = exp(-d · kinv0 · (h_c - hU)/(hB - hU))` (TR: `k = -d_clutter/ln(1 - r) · (hB - hU)/(h_c - hU)`);
- else (`hB <= hU < h_c`, BS inside the clutter below the UT): `p = exp(-d · kinv0)` (the SL/DL rule). Reason:
  the TR formula is written for a BS above the UT (the high-BS deployments have hB 8 m, above the clutter);
  applied literally here it gives `k < 0`, hence
  `p >= 1` (limited to 1), for `hB < hU`, and `k = 0`, hence `p = 0`, for `hB = hU`. With both ends in the clutter
  the clutter blocks the path as in the low-BS sub-scenarios, so their rule is the consistent fallback.
`r = 0` gives `kinv0 = 0` and p = 1.

SMa (TR v19), `d > 10`, defaults `d_clutter = 30`, `r_com = 0.02`, `r_res = 0.18`, `r_veg = r_clutter` (default 0),
heights `h_com = 20`, `h_res = 8`, `h_veg = 15` (fixed); `h_clutter` is ignored:
```
if hB <= hU:  p = 1
else:  kinv_x = -ln(1 - r_x)/d_clutter · max(h_x - hU, 0)/(hB - hU)       x = com, res, veg
       p = exp(-d · (kinv_com + kinv_res + kinv_veg))
```
(The TR has `max(., 0)` only for the residential term; for `hU < 15 m`, the TR range, the results are equal.)

#### 3.5.3 Soft LOS state (TR eq. 7.6-18)

```
F(p)            = probit(p) = sqrt(2) · erfinv(2p - 1)          p = p_LOS(l) after the [0, 1] limit
lambda_k        = c / fc(k)                                      wavelength [m]
los_soft(l, k)  = 0.5 + atan( sqrt(20 / lambda_k) · (rnd_g(l) + F(p)) ) / pi
```
`p = 1` gives `F = +Inf` and `los_soft = 1`; `p = 0` gives `F = -Inf` and `los_soft = 0` (also for any finite
`rnd_g`). `los_soft > 0.5` exactly when `rnd_g > -F(p)`, which has probability `p` for `rnd_g ~ N(0,1)`.

#### 3.5.4 Inverse normal CDF (file-local in `src/los_probability.cpp`)

`p <= 0`: `-Inf`; `p >= 1`: `+Inf`; `p > 0.5`: `F(p) = -F(1 - p)` (symmetric evaluation; `1 - p` is exact for
`p >= 0.5`). The remaining `p` in `(0, 0.5]` are evaluated in the lower tail, by Acklam's approximation with
coefficients

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
p_low <= p <= 0.5:  q = p - 0.5; r = q²;     x = (((((a0 r + a1) r + a2) r + a3) r + a4) r + a5) q / (((((b0 r + b1) r + b2) r + b3) r + b4) r + 1)
```
followed by one Halley step: `e = 0.5·erfc(-x/sqrt(2)) - p`, `u = e·sqrt(2·pi)·exp(x²/2)`,
`x = x - u/(1 + x·u/2)` (`std::erfc`). Accuracy target (what `los_soft` needs, with margin): `abs(x - F(p)) <=
1e-10 · max(1, abs(F(p)))` for `p` in `[1e-12, 1 - 1e-12]`, `F` the exact inverse of `Phi(x) = 0.5·erfc(-x/sqrt(2))`
(Acklam alone reaches about 1e-9 relative; with the Halley step in the lower tail, where `0.5·erfc(-x/sqrt(2))`
resolves `p`, the error is below about 4e-16 · max(1, abs(F)) over the whole range). Reason for the symmetric
evaluation: for `p` above about `1 - 1e-9`, `0.5·erfc(-x/sqrt(2))` rounds to `p` itself, so a Halley step in the
upper tail does nothing and Acklam's error remains (7e-9 at `1 - 1e-12`, ten times the target). The probit has one
user and is a lambda or an anonymous-namespace function of `los_probability.cpp`; it is tested through
`los_probability` (5.5).

#### 3.5.5 Documentation block (must state)

Every model with its formula and source (TR 38.901 Table 7.4.2-1, TR 38.811 via QuaDRiGa, QuaDRiGa); the defaults of
`d_clutter`, `r_clutter`, `h_clutter`; the edge rules; the [0, 1] limit; the soft LOS formula; that the LOS state draw
itself and its correlation distance (`LOS_state_lambda`) are the caller's; that for O2I UTs `d2d_out = d2D - d2D-in`
(from `o2i_loss`, run first), clamped at 0 by the caller (negative distances are rejected); that the
`LOS_state_lambda` of the InF files is `d_clutter / 2` of the TR defaults and does not follow a `d_clutter` argument
(the caller passes its own correlation distance when it changes `d_clutter`).

### 3.6 `o2i_loss`

Source: TR 7.4.3.1 (Tables 7.4.3-1 to 7.4.3-3), 7.4.3.2, 7.6.3.3; the TR models of `gen_o2i_loss.m` (lines 110 to
308).

Scope: the six models of TR 38.901 clause 7.4.3 and the TR rule for the indoor distance (1.2). The indoor part of
the 3D distance, which QuaDRiGa's legacy path-loss distance needs, is computed by `path_loss` from the indoor 2D
distance (3.3.2).

#### 3.6.1 Validation (in this order)

1. `fc` (3.0.2).
2. `rnd`: 4 columns: `Input 'rnd' must have 4 columns.`; columns 0, 1, 2 in [0, 1]:
   `Uniform columns of 'rnd' must be in the range from 0 to 1.`; column 3 finite:
   `Column 3 of 'rnd' must be finite.` `n_ut = rnd.n_rows` (0 allowed).
3. `models` non-empty (`Input 'models' cannot be empty.`); each name one of the 6 of the header
   (`Model '<name>' is not supported.`).
4. `weights`: as many elements as `models` (`Number of elements in 'weights' does not match number of models.`);
   finite and >= 0 (`Input 'weights' must be finite and >= 0.`); `abs(sum - 1) <= 1e-6`
   (`Input 'weights' must sum to 1.`).
5. `d2d_in_max` finite and >= 0: `Input 'd2d_in_max' must be finite and >= 0.`

#### 3.6.2 Algorithm (per UT `u`, `f` in GHz per frequency)

1. Model choice: `Ccum_j = (w_0 + ... + w_j) / sum(w)`; model `j*` = smallest `j` with `rnd(u, 0) < Ccum_j`;
   if there is none (u >= the last cumulative value), the last `j` with `w_j > 0`. (A weight of 0 is never chosen.)
2. Indoor distance:
   - `legacy_6GHz`: `d2din = rnd(u, 1) · d2d_in_max` (one uniform, Table 7.4.3-3).
   - `low`, `high`, `low_A`: `d2din = min(rnd(u, 1), rnd(u, 2)) · d2d_in_max` (clause 7.4.3.1).
   - `car`, `car_metal`: `d2din = 0`.
3. Loss per frequency:
   ```
   L_glass = 2 + 0.2 f;  L_IRR = 25.4 + 0.11 f;  L_concrete = 5 + 4 f;  L_plywood = 1.03 + 0.17 f
   tw(p1, L1, p2, L2) = 5 - 10·lg(p1·10^(-L1/10) + p2·10^(-L2/10))
   ```
   | Model | PL_tw | PL_in | deviation |
   |---|---|---|---|
   | `low` | `tw(0.3, L_glass, 0.7, L_concrete)` | `0.5·d2din` | `4.4 · rnd(u, 3)` |
   | `high` | `tw(0.7, L_IRR, 0.3, L_concrete)` | `0.5·d2din` | `6.5 · rnd(u, 3)` |
   | `low_A` | `tw(0.3, L_glass, 0.7, L_plywood)` | `0.5·d2din` | `4.4 · rnd(u, 3)` |
   | `legacy_6GHz` | 20 | `0.5·d2din` | 0 |
   | `car` | 0 | 0 | `9 + 5 · rnd(u, 3)` |
   | `car_metal` | 0 | 0 | `20 + 5 · rnd(u, 3)` |

   `rnd(u, 3)` is N(0,1) as in the TR (`gen_o2i_loss.m` draws the deviation from a uniform variable, so QuaDRiGa's
   O2I losses are not reproduced). `loss(u, k) = PL_tw + PL_in + deviation`.
4. `d2d_in(u) = d2din`; `model_index(u) = j*` (0-based).

#### 3.6.3 Documentation block (must state)

The 6 models with their table formulas (`high` with the v19 IRR glass 25.4 + 0.11 f), the column meaning of `rnd`,
the model choice rule, the d2D-in rules, that the caller draws `rnd` with `sos_rand` / `sos_randn` at the TR
correlation distances (building type 50 m, d2D-in 25 m, sigma_P 10 m, clause 7.6.3.3) and adds the result to the
path loss of the indoor UT; that `d2d_in` gives `d2d_out` for `los_probability` and is the input of the legacy
outdoor distance of `path_loss`; RMa uses `low` only, InF `high` only; the backward compatible model is
`legacy_6GHz` with `O2I`-file `SF_sigma = 7` (the UMa/UMi O2I files already have 7 dB).

### 3.7 The moved functions

`conf_file_read`, `scenpar_validate`, `conf_file_list`, `conf_file_write`: no change of behaviour except the six new
keys (4.1), the V/S term in the DS overview line of `conf_file_write` (4.1) and the one new cross-key check of 4.2.
Declarations move to `quadriga_gbsm.hpp`; the section of their documentation blocks (C++, MEX, Python) becomes
"Geometry-based stochastic model"; the key counts in the documentation change from 165 to 171 numeric keys ("138 keys
of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0"), 166 to 172 MEX fields / Python dict entries; Python moves them
from `quadriga_lib.tools` to `quadriga_lib.gbsm`.


## 4. Conf-key changes

### 4.1 New numeric keys (Run 1)

Appended to the key table `qd_scenpar_keys[]` in `src/qd_scenpar.cpp` after `PolVar_sigma`, in this order, under the
comment line given below. Defaults reproduce today's behaviour; class `QD_NONNEG` (finite and >= 0); `row = col = -1`.
Writer section (`sec`) and paragraph (`par`) as below; the writer sorts by paragraph, then table order, so the keys
land after the existing keys of the same paragraph.

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
| `DS_VS` | Reference V/S of the InF hall | `lsp_generate` (3.0.4) |
| `DS_VS_a`, `DS_VS_b` | `mu_lgDS = DS_mu + lg(a · V/S + b)` | `lsp_generate` |

After Run 1: `n_key = 171`; MEX struct and Python dict of `scenpar` have 172 entries (with `SubpathMethod`); the
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
Python) adds to the overview item: "the DS line has the term log10( a * V/S + b ) when DS_VS > 0".

### 4.2 Validation addition (Run 1)

In `qd_scenpar_check` (after the `NF_alpha`/`NF_beta` check), a cross-key check:

```cpp
const arma::uword i_vs = qd_scenpar_find("DS_VS"), i_vsa = qd_scenpar_find("DS_VS_a"), i_vsb = qd_scenpar_find("DS_VS_b");
if (val[i_vs] > 0.0 && val[i_vsa] == 0.0 && val[i_vsb] == 0.0)
    throw std::invalid_argument(where(line == nullptr ? 0 : line[i_vs]) +
                                "Values of 'DS_VS_a' and 'DS_VS_b' must not both be 0 if 'DS_VS' is > 0.");
```

The `scenpar_validate` documentation lists the 6 keys under "Finite and >= 0" and the new rule under the cross-key
rules (C++, MEX, Python blocks).

### 4.3 New `PL_` keys (Run 3a)

No interface change (`plpar` is an open map). Behaviour in 3.3.4 and 3.3.6.

| Key | Model | Meaning | Default when absent |
|---|---|---|---|
| `PL_hE_rand` | dual_slope, nlos | non-zero: UMa random hE of note 1 from `rnd_he` | 0 (fixed `PL_hE`) |
| `PL_LOSmax` | nlos | 0: no maximum with the LOS term | 1 |
| `PL_Ax`, `PL_Bx`, `PL_Cx` | nlos | extra term `Ax·lg d3D + Bx + Cx·lg f` inside the maximum; active if one of them is present | inactive |

Writer comments: Run 3a adds these rows to `qd_conf_pl_comments` in `src/qd_scenpar.cpp` (before the last row):

```cpp
    {"hE_rand", "", "1 = random environment height (TR 38.901 Table 7.4.1-1, note 1); 0 = fixed hE"},
    {"LOSmax", "", "0 = no maximum with the LOS-PL; 1 = PL = max( PL_LOS, PLn )"},
    {"Ax", "", "TX-RX 3D dist.-dep. of the additional PL term in the maximum [dB/log10(m)]"},
    {"Bx", "", "Reference PL of the additional term in [dB] @ 1 GHz, 1 m TX-RX dist."},
    {"Cx", "", "Freq.-dep. of the additional PL term in [dB/log10(GHz)]"},
```

Tests of the writer rows: the case "path-loss comments and formulas depend on the model" of
`test_conf_file_write.cpp` gets one line per new key (5 keys) checking its comment.

The formula blocks `qd_conf_pl_formulas` are not changed (the written files of existing tests stay the same).

### 4.4 Values in the 23 v19 files

Lines are written in the style of the files (key name, `=`, value aligned with the neighbouring lines, `%` comment).
The names below are the scenario names without the prefix `3GPP_38.901_v19_`.

#### 4.4.1 Run 1: new numeric keys, the InF delay spread and the indoor TOA bound

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
| InF_LOS | "Large scale distributions", DS paragraph | `DS_mu = -9.35`, `DS_VS = 4.27`, `DS_VS_a = 26`, `DS_VS_b = 14` |
| InF_NLOS_SL, InF_NLOS_DH | same | `DS_mu = -9.44`, `DS_VS = 4`, `DS_VS_a = 30`, `DS_VS_b = 32` |
| InF_NLOS_DL, InF_NLOS_SH | same | `DS_mu = -9.44`, `DS_VS = 4.55`, `DS_VS_a = 30`, `DS_VS_b = 32` |
| InF_NLOS_SL, InF_NLOS_DH | "Absolute time of arrival offset", after `absTOA_lambda` | `absTOA_Lmax = 120` |
| InF_NLOS_DL, InF_NLOS_SH | same | `absTOA_Lmax = 300` |
| Indoor_NLOS | same | `absTOA_Lmax = 120` |

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
  % (120 x 60 x 10 m: 4; 300 x 150 x 10 m: 50/11 = 4.55); set DS_VS for another hall.
  ```
  InF_LOS: `% (LOS: the average of the two calibration halls, 47/11 = 4.27.)` as fourth line.
- `absTOA_Lmax`: the value line with the comment `% largest dimension L of the hall [m]; excess delay <= 2L/c`;
  the line `% L depends on the deployment and is not set here: absTOA_Lmax = 0 means no bound.` is replaced by
  `% L = max(length, width, height) of the calibration hall of TR 7.8.4 (120 x 60 x 10 m: 120 m;` and
  `% 300 x 150 x 10 m: 300 m); set absTOA_Lmax for another hall.` (InF), or
  `% L of the office of TR Table 7.2-2 (120 x 50 x 3 m): 120 m; set absTOA_Lmax for another room.` (Indoor).

`absTOA_Lmax` follows TR clause 7.6.9, which bounds the NLOS excess delay by `2L/c` for InF and InH with `L` the
largest dimension of the factory hall or office room; the files use the halls they already use for `DS_VS` (SL and
DH the small hall, DL and SH the big hall) and the office of the InH evaluation parameters. Without the bound the
lognormal tail exceeds the TR limit (e.g. InF small hall: `10^(-7.5 + 0.4·x)` s exceeds `2·120 m/c = 0.8 µs` for
`x > 3.5`; InH: `10^(-8.6 + 0.1·x)` s never reaches it for `abs(x) < 9`).

`DS_VS` of InF LOS is 47/11, the average V/S of the two calibration halls, because the one InF LOS file serves all
InF sub-scenarios; each InF NLOS file uses the V/S of its own hall (SL, DH: 4; DL, SH: 50/11). Today's `DS_mu = -7.2535` of InF LOS
is the mean of lgDS of the two halls; with V/S = 47/11 the median becomes -7.2528 (a change of 0.0007 in lg, 0.16 % in DS).
The files write the V/S values rounded by the precision rule of 1.2 (47/11 as 4.27, 50/11 as 4.55; the comments give
the fractions): mu_lgDS changes by -2.5e-4 (InF LOS, -0.06 % in DS) and +3.5e-4 (DL, SH, +0.08 % in DS), within the
0.1 % of the rule; with 4.27 the InF LOS median is -7.2530.

#### 4.4.2 Run 3a: path-loss keys and constants

The derived path-loss constants follow the precision rule of 1.2: 4 significant digits, more only where 4 digits
would change the loss by more than 0.01 dB. Bounds of the change against the exact constants (`c = 3e8`, the h and W
of the files), evaluated in the `dual_slope` form of 3.3.4 over the TR ranges (UMa, UMi: hUT 1.5 to 22.5 m, 0.5 to
100 GHz, d2D up to 5 km; RMa: hBS 10 to 150 m, hUT 1 to 10 m, 0.5 to 30 GHz; SMa: hBS 25 to 35 m, hUT 1 to 14 m,
0.5 to 37 GHz; RMa and SMa d2D up to 10 km):
- UMa, UMi: `PL_E = 13.33` instead of 40/3 changes the loss beyond the breakpoint by `(A2 - A1)·lg((40/3)/13.33)`,
  at most 2.1e-3 dB (UMa 1.95e-3, UMi 2.06e-3).
- InH and InF NLOS files: no change (`A1 = A2`, the breakpoint has no effect).
- RMa LOS term: `A1 = 20.48`, `B = 31.74`, `D = 0.001398`, `E = 20.94`: at most 8.2e-3 dB (8.1e-3 dB at the test
  points of 5.3); RMa NLOS term `Bn = 31.75`: 5.0e-3 dB.
- SMa LOS term: `A1 = 21.574` (5 digits: 21.57 would change the loss by up to 0.023 dB at 10 km), `B = 30.13`,
  `E = 20.94`: at most 6.6e-3 dB.
Against the values of the files today (13.34, 20.944, 0.0014, ...), the path loss changes by up to 0.014 dB (RMa,
mostly `PL_D` 0.0014 instead of 0.001398 at 10 km; UMa, UMi 0.006 dB, SMa 0.005 dB).

| Files | Change |
|---|---|
| Indoor_NLOS, InF_NLOS_DH, _DL, _SH, _SL, UMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I, UMi_LOS, _LOS_O2I, _NLOS, _NLOS_O2I (13 files) | `PL_E = 13.33` (was 13.34, 40/3 rounded upwards); value comment `% Breakpoint scaling factor` unchanged |
| RMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I, SMa_LOS, _LOS_O2I | `PL_E = 20.94` (was 20.944); comment `E = 2 pi / 0.3 = 20 pi / 3 = 20.94` |
| RMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I | `PL_A1 = 20.48`, `PL_B = 31.74`, `PL_D = 0.001398` (were 20.478, 31.741, 0.0014) |
| RMa_NLOS, _NLOS_O2I | `PL_Bn = 31.75` (was 31.755) |
| SMa_LOS, _LOS_O2I | `PL_A1 = 21.574` (unchanged, 5 digits as above), `PL_B = 30.13` (was 30.133); `PL_D = 0.002` exact |
| SMa_NLOS, _NLOS_O2I | remove `PL_A1`, `PL_A2`, `PL_B`, `PL_C`, `PL_E`, `PL_hE`; add `PL_LOSmax = 0` |
| UMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I | add `PL_hE_rand = 1` after `PL_hE` |
| InF_NLOS_DL | add `PL_Ax = 25.5`, `PL_Bx = 33`, `PL_Cx = 20` |

Comment changes in the path-loss sections:
- UMa (4 files): `hE = 1 m (note 1; the random hE for hUT > 13 m is not modelled)` becomes
  `hE = 1 m or random for hUT >= 13 m (note 1; PL_hE_rand = 1, drawn from column 10 of lsf_draw)`;
  `E = 4 / 0.3 = 13.34 (rounded)` becomes `E = 4 / 0.3 = 40/3 = 13.33` (in the line `% Break point of note 1: ...`). The
  UMi, Indoor_NLOS and InF_NLOS files have no such explanatory line; no line is added there (only the value
  changes).
- RMa (4 files), SMa_LOS, SMa_LOS_O2I: `E = 2 pi / 0.3 = 20.944` becomes `E = 2 pi / 0.3 = 20 pi / 3 = 20.94` (in the
  line `% Break point of note 5: ...`); a line
  `% The constants are evaluated for h and W given above and rounded to 4 significant digits (change < 0.01 dB).`
  (SMa_LOS, SMa_LOS_O2I: `... rounded to 4 significant digits, A1 to 5 (change < 0.01 dB).`); and in RMa_LOS, _LOS_O2I,
  _NLOS, _NLOS_O2I, SMa_LOS, _LOS_O2I after the QuaDRiGa formula block
  `% dual_slope evaluates PL2 with the 3D break-point distance sqrt( dBP^2 + (hBS-hUT)^2 ); the TR form with dBP`
  `% differs by less than 0.035 dB.`
- SMa_NLOS, SMa_NLOS_O2I: replace the two lines starting `% The TR has no max( PL_LOS, PLn ) for SMa NLOS.` by
  `% The TR has no max( PL_LOS, PLn ) for SMa NLOS: PL_LOSmax = 0 switches the LOS term off.`; the QuaDRiGa
  dual-slope formula block (`% Formula for the QuaDRiGa dual-slope (LOS) model:` and its 4 lines) and the line
  `% Break point of note 5: ...` are removed from these files (`PL_E` and `PL_hE` are removed).
- InF_NLOS_DL: replace `% The TR takes the maximum of three terms ... below about 26 m.` by
  `% The TR takes the maximum of three terms for InF-DL: max( PL, PL_LOS, PL_InF-SL ). The InF-SL term is the
  % additional term PLx = Ax * log10( d3D ) + Bx + Cx * log10( fGHz ) of the QuaDRiGa model "nlos".`
- Optional NLOS formulas of Table 7.4.1-1, in UMa_NLOS, UMi_NLOS, Indoor_NLOS after the TR formula lines:
  ```
  % Optional NLOS formula of Table 7.4.1-1 (UMa): PL = 32.4 + 20 log10(fc) + 30 log10(d3D), SF std 7.8 dB.
  % It is used with PL_model = logdist, PL_A = 30, PL_B = 32.4, PL_C = 20, PL_SF = 7.8.
  ```
  UMi: `31.9 log10(d3D)`, SF 8.2 (`PL_A = 31.9`, `PL_SF = 8.2`); Indoor: `PL' = 32.4 + 20 log10(fc) + 31.9
  log10(d3D)`, SF 8.29, used in `max(PL_LOS, PL')` (model `nlos` with `PL_An = 31.9`, `PL_Bn = 32.4`, `PL_Cn = 20`,
  `PL_sig = 8.29`).

After Run 3a every v19 file reads without ignored keys, passes the value checks and keeps a positive definite
correlation matrix (existing test).


## 5. Catch2 tests

### 5.0 Conventions for all new test files

- One file per source file: `tests/catch2_tests/test_gbsm_phases.cpp` (`lsf_init_phases` and `lsf_draw`, one
  fixture block), `test_path_loss.cpp`, `test_lsp_generate.cpp`, `test_los_probability.cpp`, `test_o2i_loss.cpp`.
  No test of the internal header: every helper is covered through a public function with an independent oracle
  (5.7).
- Header comment as SPEC_CPP_API (what is pinned, oracles, conventions). Include `"quadriga_gbsm.hpp"` (and
  `"quadriga_sos.hpp"` where SOS functions provide supporting data).
- Diagnostics: `#include "qd_test_diag.hpp"` and `QD_TEST_DIAG_SWITCH("QD_DIAG_GBSM")`; Run 2a adds the line
  `//   QD_DIAG_GBSM   large-scale model tests (test_gbsm_phases.cpp, test_path_loss.cpp, test_lsp_generate.cpp, ...)` to
  the list in `tests/catch2_tests/qd_test_diag.hpp`. Tables in CSV form with `# diag:` title. (The `QD_TEST_VERBOSE`
  define of SPEC_CPP_API is not used; the test suite has moved to the area switch.)
- Test case names `"<function> - <statement>"`, unique in the suite; one behaviour per case; one validation case
  per function named `"<function> - input validation"` with a valid baseline (`CHECK_NOTHROW`) and one `SECTION`
  per check of the validation list. Messages are checked with `CHECK_THROWS_MATCHES(expr, std::invalid_argument,
  Catch::Matchers::Message("..."))`; no test-side try/catch helper.
- Oracles are typed into the test file: TR 38.901 v19.5.0 formulas and tables (cited by clause, table and equation
  number), the formulas of this spec (written out, not cited by section), a scalar sum-of-sinusoids
  `ref_sos(F, phi, pos) = sqrt(2/N) · sum_n cos(2·pi·(F(n,0)·x + F(n,1)·y + F(n,2)·z) + phi(n))`, the uniform map
  `0.5·erfc(-v/sqrt(2))`, a test-side 8x8 Cholesky. Supporting tasks use library calls, not test-side
  re-implementations: `sos_preset` (frequency and ACF tables), `sos_acfi` (linear interpolation of an ACF table),
  `interp_1D` (linear interpolation of the typed zenith and 38.811 tables, which stay typed in the tests as oracle
  data), `sos_init_phases`, `conf_file_read` (the files under test), `lsf_init_phases` / `lsf_draw` where a test of
  another function needs phases or random values.
- Never an oracle: the function under test or `quadriga_gbsm_helpers.hpp`.
- Random inputs of statistical tests come from `std::mt19937_64` with fixed seeds and test-side transforms
  (Box-Muller for Normal values); `std::normal_distribution` is not used (not portable).
- Tolerances: exact-by-construction double results 1e-12 (absolute for O(1) values, relative for others); dB values
  1e-9 where long chains of `lg` and constants enter; statistical tolerances = 5 standard errors plus the model
  approximation error, stated in a comment.
- Tolerances of values produced by `lsf_draw` (AVX2 kernel with single-precision sin/cos, 3.0.1), with the
  reason in a comment: Normal columns 2e-5 absolute for 256 sinusoids, scaled with `sqrt(n_coeff / 256)` for other
  presets (1.5e-5 for 128, 4e-5 for 1024), also for the dual-mobility columns (two kernel errors divided by at
  least sqrt(2)); Uniform columns 1e-5 (the slope of the uniform map is at most 0.4); zero
  sample means through `lsf_draw` 2e-5. Checks on phases with a test-side `ref_sos` (no `lsf_draw` involved) stay
  1e-12. Equality checks that compare the same evaluation stay exact: z independence in 2D mode, independence of
  the other links, determinism, swap reciprocity, `floor_id` all 0 equals `nullptr`, single mobility ignores
  `tx_pos`. `lsp_generate`, `path_loss`, `los_probability`, `o2i_loss` take typed random values and keep the
  double tolerances above.
- Test files contain no internal references (9.1): no IDs of planning decisions or review findings, no section numbers
  of this spec, no `md/` or local paths, no QuaDRiGa-NG line numbers; comments cite TR 38.901 clauses, tables and
  equations or state the rule.
- Determinism: a repeated call gives identical results (one check per file).

### 5.1 `test_gbsm_phases.cpp`: `lsf_init_phases` cases

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
| input validation | one section per check of 3.1: seed -2 (`Input 'seed' must be >= -1.`); unknown preset (`Input 'preset' is not a known preset.`); `DS_lambda` -1 and `SC_lambda` Inf (the `scenpar_validate` message passes through); an unknown key; `rx_pos` 2 columns; `rx_pos` NaN; `tx_pos` 2 columns; `tx_pos` 2 rows for 5 links; dual mobility with positions, bias removal, no `tx_pos`; baseline: the same call without bias removal does not throw | — | — |

### 5.2 `test_gbsm_phases.cpp`: `lsf_draw` cases

Tolerances of `lsf_draw` values follow 5.0 (AVX2 kernel); the reason is stated in a comment of each case.

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| single mobility values equal the sum of sinusoids of each generator | 7 links with distinct coordinates (negative, up to 300 m, z != 0), distinct lambdas (as in 5.1); column g equals `ref_sos(F_g, slice0.col(g), (x, y, 0))`, columns 9, 10 the uniform map of it; column 10 uses `SF_lambda` | `ref_sos`, `sos_preset(.., lambda_g)` | 2e-5 (Normal), 1e-5 (Uniform) |
| 3D mode uses the z coordinate | `horizontal = false`: `ref_sos(.., (x, y, z))` | as above | 2e-5 / 1e-5 |
| other presets | Exp128 and Gauss1024: column 0 equals `ref_sos` | as above | 1.5e-5, 4e-5 |
| 2D mode values do not depend on z | changing all z leaves `rnd` unchanged | — | exact |
| single mobility ignores tx_pos | two different `tx_pos` give identical `rnd` | — | exact |
| a link does not depend on the other links | links evaluated one by one equal the joint call | — | exact |
| dual mobility values follow the two-device formulas | 6 links incl. one with Tx = Rx and one with distance > 5 lambda; self-mirrored columns `(ref_sos(F_g, P0_g, R) + ref_sos(F_g, P0_g, T)) / sqrt(2 + 2·A_g(D))`; columns 3..6 `(ref_sos(F_g, P0_g, R) + ref_sos(F_m, P0_m, T)) / sqrt(2)` (no further mixing of the pair); 9, 10 the uniform map | typed formulas, `ref_sos`, the `sos_preset` ACF table interpolated by `sos_acfi`; floors 3 and 259 mixed with floor 0 in the same 6 links: the typed offsets of 3.2.1 added to the phases of both sides | 2e-5 (0..8), 1e-5 (9, 10) |
| dual mobility keeps unit variance and correlates the pairs by the ACF at the Tx-Rx distance | lambdas ASD 10, ASA 15, ESD 10, ESA 20 m; the 100 x 100 Rx grid of the marginal case (50 m spacing), Tx = Rx + (D, 0, 0) for D = 5, 10, 20 m (one call per D): sample variance of columns 3, 4, 5, 6 within 0.07 of 1; sample correlation of (3, 4) within 0.08 of `(A_3(D) + A_4(D))/2` and of (5, 6) within 0.08 of `(A_5(D) + A_6(D))/2`; D = 0 (Tx = Rx): column 3 equals column 4 and column 5 equals column 6 | unit variance; `sos_preset` ACF tables, `sos_acfi` | variance 0.07 = 5 standard errors (`sqrt(2/n)` = 0.014); correlation 0.08 = 5 standard errors (<= 0.01) + SOS approximation error (0.03); D = 0 exact |
| swapping Tx and Rx swaps ASD with ASA and ESD with ESA | dual mobility, unequal lambdas of ASD/ASA and ESD/ESA, phases with and without bias removal; both calls use one phase cube (computed once from the original positions): `rnd(tx=A, rx=B)` columns 0, 1, 2, 7, 8, 9, 10 equal `rnd(tx=B, rx=A)`, 3 <-> 4 and 5 <-> 6 swapped | reciprocity (same evaluations in swapped roles) | exact |
| floor 0 equals no floor ID and equal IDs share one field | `floor_id = 0` gives the values of `nullptr` (exact); links of floor 3 and floor 259 equal `ref_sos` with the phases plus the offsets `2·pi·k·n/N + psi_b(n)` typed in the test (`k = q mod N`, `b = q div N`; `psi_1(n) = 2·pi·((x_n >> 11)·2^-53)` with `x_n` the n-th output of `std::mt19937_64 gen(1)`) | typed offset rule | exact; 2e-5 |
| fields of different floors are uncorrelated at equal positions | lambda 10 m, 100 x 100 grid with 50 m spacing (n = 10^4, ACF at 5 lambda below 0.02), the same positions on every floor: sample correlation of every Normal column between floors (0, 1), (1, 5), (0, 255) below 0.05 in magnitude; between (0, 256) below 0.25 | zero within a block of N floor IDs, `1/sqrt(2N) = 0.044` between blocks | 0.05 = 5 standard errors of 1/sqrt(n); 0.25 = 5.6 · 0.044 |
| marginal distributions | same grid, floor 0: columns 0..8 `abs(mean) < 0.05`, `abs(std - 1) < 0.05`; columns 9, 10 in [0, 1], `abs(mean - 0.5) < 0.015`, fraction below 0.1 within 0.015 of 0.1 | N(0,1), U(0,1) | 5 standard errors (0.01, 0.003, 0.003) |
| the correlation follows the preset ACF | lambda 10 m, base grid as above, offsets 5, 10, 20 m in 4 directions: sample correlation within 0.1 of the `sos_preset` ACF at that distance (Comb256 and Exp256) | preset ACF table | 0.1 >= 5 standard errors (0.05) + directional SOS error (up to 0.045): the sample correlation converges to the ACF of the sinusoids along the offset direction, not to the table ACF (e.g. Comb256 at 10 m: 0.408 along x, 0.393 along y, against 0.368; Exp256 at 5 m along y: 0.572 against 0.607); stated in a comment |
| sizes, empty input and pre-sized output | `n_link = 0` gives `[0, 11]` without an exception (single and dual mobility); a pre-sized `[n_link, 11]` output with stale content is overwritten; a wrong size is resized | — | exact |
| input validation | sections: `phase` `[128, 11, 2]` with Comb256, `[256, 10, 2]`, `[256, 11, 1]`, NaN entry; unknown preset (`Input 'preset' is not a known preset.`); `rx_pos` 2 columns; Inf in `tx_pos`; `tx_pos` 2 rows for 3 links; `floor_id` length; negative `XPR_lambda` (`scenpar_validate` message) | — | — |

### 5.3 `test_path_loss.cpp`

Range limit of the TR-oracle points: only points inside the TR applicability range of each formula are compared
with the rounding tolerances of 4.4.2 (d2D up to 5 km for UMa, UMi, RMa NLOS, SMa NLOS; 10 km for RMa LOS and SMa
LOS). A break-point point (`0.5·dBP`, `2·dBP`) beyond this range is skipped, because the rounded constants are only
accurate to the stated bound inside the range.

TR oracle: Table 7.4.1-1 typed with `c = 3e8`, RMa `h = 5, W = 20`, SMa `h = 10, W = 10`, `hE = 1` for UMa/UMi (note 1
with C = 0), RMa/SMa LOS `PL2 = PL1(dBP) + 40·lg(d3D/dBP)` with the 2D breakpoint distance, InF-DL `max(PL, PL_LOS,
PL_InF-SL)`, SMa NLOS without max, the shadow-fading std of the table (RMa/SMa LOS 4 or 6 by the side of `dBP`). The
SF oracle of the O2I files is the O2I shadow fading of Table 7.5-6 (7 dB UMa/UMi, 8 dB RMa/SMa, equal to `SF_sigma` of
the files; they have no `sig` keys), not the outdoor value of Table 7.4.1-1. Points are chosen at `0.5·dBP` and
`2·dBP` (never at the breakpoint itself).

Rounded constants: the files write the derived constants rounded by the precision rule of 1.2 (4.4.2), the oracle
uses the exact TR expressions. The tolerance of each file is the bound of that rounding over the TR ranges (4.4.2),
written out with its reason in a comment of the test: UMa, UMi `tol_r = 2.1e-3 dB` (`PL_E = 13.33` for 40/3:
`19·lg((40/3)/13.33)`); RMa `tol_r = 8.2e-3 dB` (A1, B, D, E of the LOS term; the NLOS term with `Bn = 31.75`
differs by 5e-3 dB, so the maximum stays within the bound); SMa LOS `tol_r = 6.6e-3 dB`; InH, InF and SMa NLOS
`tol_r = 1e-9 dB` (no rounded constant reaches the result there). The O2I files take the bound of their outdoor file.

RMa and SMa beyond the breakpoint: the library evaluates PL2 with the 3D breakpoint distance
`dBP3 = sqrt(dBP² + (hBS - hUT)²)` (3.3.4), the TR with the 2D one. For these points (the LOS term of RMa LOS, RMa
NLOS and SMa LOS, each also O2I, at `d2D > dBP`) the test asserts (a) `abs(pl - PL_TR) <= 0.035 dB + tol_r` against
the TR formula and (b) `pl` equal to the TR formula with PL2 replaced by `PL1(dBP3) + 40·lg(d3D/dBP3)` to `tol_r` (for RMa
NLOS inside the maximum). The tolerance of (a) is the worst case of the difference over the TR ranges, given with
its ranges in 3.3.4 (at the test points the difference is at most 4.1e-3 dB), plus the rounding bound `tol_r`. A
comment in the test states this reason with the bound and the ranges written out; (b) separates the breakpoint form
from the rounding of the constants.

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| the v19 files reproduce TR 38.901 Table 7.4.1-1 | for every LOS, NLOS and O2I file: `pl` and `sf_sigma` at the points: UMa hBS 25, UMi hBS 10: hUT {1.5, 12}, fc {3.5, 28} GHz, d2D {15, 0.5 dBP', 2 dBP' if <= 5 km, 1000, 4500}; RMa hBS {35, 25}, hUT {1.5, 5}, fc {0.9, 3.5, 6}, d2D {50, 0.5 dBP, 2 dBP (LOS <= 10 km, NLOS <= 5 km), 4000}; SMa hBS {25, 35}, hUT {1.5, 4.5, 13.5}, fc {3.5, 20}, d2D {50, 0.5 dBP, 2 dBP, 3000}; InH hBS 3, hUT 1, fc {3.5, 28, 60}, d2D {1, 5, 20, 100}; InF hBS {1.5, 8}, hUT 1.5, fc {3.5, 28}, d2D {1, 5, 10, 30, 100, 500}. O2I files equal the outdoor TR formula at the geometric distance, `sf_sigma` the O2I value. A diagnostic counts the InF-DL points where the InF-SL term wins (> 0 asserted) | typed TR formulas | `tol_r` of the file (rounded constants, see above); RMa/SMa LOS terms beyond dBP: 0.035 dB + `tol_r` against the TR, `tol_r` against the 3D-breakpoint form (see above); `sf_sigma` exact |
| O2I distances: geometric distance without d2d_in, outdoor distance with it | UMa NLOS file, hand-built links: the call without `d2d_in` equals the TR formula at the geometric distance (covered above); with `d2d_in` the typed formula at the outdoor distance of the geometry rule (`din = min(d2d_in, d2)`, `d3in = d3·din/d2`, outdoor `d3 - d3in` and `d2·(1 - d3in/d3)`); `d2d_in > d2D`; the 10 m minimum outdoor distance (d3D 15 m, `d2d_in` 12 m); `d2D = 0` with `d2d_in > 0`; `d3D = 0` (indoor fraction 0); `d2d_in = 0` equals the call without `d2d_in`, also at `d2D = 0` | typed legacy distance rule | 1e-12 dB |
| UMa random hE follows the probability of TR note 1 | UMa LOS file, fc 0.05 GHz, hBS 25, hUT 20, d2D 400 m, K = 10000 links with `rnd_he = (k + 0.5)/K`: hE of each link identified by matching `pl` with the TR LOS formula for hE in {1, 12, 15, 18} (the candidates differ by > 0.01 dB, asserted first; more than twice the rounding bound 2.1e-3 dB of `PL_E`, so the nearest candidate is unique); count of hE = 1 equals `K/(1 + C)` within 1; counts of 12, 15, 18 equal `K·(1 - p1)/3` within 2; hUT 13.2 and hUT 10: all hE = 1; `rnd_he` omitted: hE = 1. Model `nlos`: the NLOS term of Table 7.4.1-1 is larger than the LOS term at most of these points, so the LOS term is observed through a plpar copy of the UMa NLOS file with `Bn = -1000` (the NLOS term never wins; the `nlos` code path with `hE_rand` is kept): with the same `rnd_he` its `pl` equals the TR LOS formula with the hE identified from the LOS file, link by link; the unmodified UMa NLOS file gives `max(PLn_TR, PL_LOS_TR(hE))` with that hE. TR UT heights of floors 7 and 8 (no 1.1·hE clamp of the UT height): hBS 25, hUT 19.5 and 22.5, fc 3.5 GHz, d2D 200 and 1000 m (both sides of d'BP = 490 m and 280 m), `rnd_he = p1 + (n - 0.5)/n · (1 - p1)` with `p1`, `n` of note 1 computed in the test, so that hE = 18 m and 21 m (h'UT = 1.5 m): `pl` equals the TR LOS formula with that hE for the UMa LOS file and for the `Bn = -1000` copy of the UMa NLOS file; the unmodified UMa NLOS file gives `max(PLn_TR, PL_LOS_TR(hE))` (the NLOS term wins there by 14 to 20 dB) | TR note 1 and Table 7.4.1-1 typed | match 2.1e-3 dB (`tol_r` of UMa: `PL_E = 13.33`); counts deterministic; floors 7 and 8: 2.1e-3 dB |
| legacy models match the get_pl formulas | one SECTION per model with distinct parameter values, 4 links x 3 frequencies: constant; logdist with and without `SF`; logdist_simple; winner_los (links with different heights incl. < 1.5 m, both branches, breakpoint from each link's own heights, `C1 != C2` so that the second branch shows `C2`, continuity at the breakpoint for `F1 = F2`); winner_nlos (fc 1.0, 1.7, 2.5 GHz); winner_pathloss with `SF`; dual_slope (defaults A2, C, D; hE clamp 1.1·hE; both sides with `hBS != hUT`, so that the 3D breakpoint distance differs from the 2D one; sig1/sig2); tripple_slope (three regions, `E2` default, dBP1 > dBP2 case); nlos (all 13 terms non-zero, G2n default, `LOSmax` 0 and 1, `Ax/Bx/Cx`, `sig`); satellite (`usePLa` 1 at 1, 20.71, 33 and 150 GHz, `usePLa` 0, 600 km height, below the horizon gives +Inf) | model formulas typed in the test; zenith-attenuation table typed | 1e-10 dB |
| distances below 0.1 m are clamped | co-located Tx and Rx give the loss at 0.1 m | typed | 1e-12 dB |
| default sf_sigma follows the LSP formula | `SF_sigma 6, SF_delta 1.5, SF_omega 0.5, SF_kappa 0.3, SF_tau -0.2, SF_beta 0.1, LSP_fc_min 6`, fc 3 and 10 GHz, logdist model without `SF`; a second scenpar with `SF_sigma 1.4, SF_beta 1` (all other SF terms 0) and links at elevations 1 deg and 10 deg: `sf_sigma` 0 at 1 deg (the terms give -0.36), `1.4 + lg(a)` at 10 deg | LSP std formula typed, including both limits at 0 | 1e-12 |
| an empty model gives zero loss | `pl == 0`, `sf_sigma` default | — | exact |
| optional outputs and pre-sized outputs | `pl` with and without `sf_sigma` equal; pre-sized outputs overwritten; `n_link = 0`; one `tx_pos` row equals repeated rows; repeated call identical | — | exact |
| input validation | sections for every check of 3.3.1 (unknown model; a missing required key for each model with required keys; empty value vector and a vector with two values (`Path-loss parameter 'B' must have one value.`); `E = 0`; `winner_los` `A1 == A2`; `d2d_in` negative and wrong length; `rnd_he` 1.5 and wrong length; `fc` empty, 0, NaN; positions; an invalid scenpar value passes the `scenpar_validate` message through) | — | — |

### 5.4 `test_lsp_generate.cpp`

The formula mechanics are pinned with custom scenpar values in which every term is non-zero and distinct (all
`_gamma`, `_epsilon`, `_zeta`, `_alpha`, `_delta`, `_kappa`, `_tau`, `_beta` terms, `_omega`, `LSP_fc_min`, the ESD
terms and floor, the V/S keys), typed against the mean/std formula written out in the test. The per-file check of
the 23 v19 files against Tables 7.5-6 to 7.5-12 stays in `test_conf_files_v19.cpp` (key level); this file adds TR
spot checks through `lsp_generate` for four files. Reference geometry: `tx = (0, 0, hBS)`,
`rx = (d2D, 0, hUT)`. Where the TR has "N/A" (KF of NLOS and O2I links) the oracle is the QuaDRiGa convention of
the files: `KF_mu = -100` dB, `KF_sigma = 0`.

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| mean and std follow the formula with every term | custom keys as above, 4 links (hUT above and below hBS, a point where the ESD floor applies, and a link at d2D 5000 m where a negative `AS_D_kappa` makes the ASD std negative before the outer limit, so `lsp_sigma` slice 3 is 0 and `lsp` slice 3 equals `10^mu`), fc {1.5, 3.5, 28} GHz with `LSP_fc_min = 2`: `lsp_mu` and `lsp_sigma` of all 8 slices; SF mean 0 | mean/std formula typed, including both limits at 0 | 1e-12 |
| TR spot checks of four v19 files | `lsp_mu`, `lsp_sigma` of all 8 slices: UMa NLOS (hBS 25, hUT 1.5 and 12, d2D 200 m, fc 3.5 and 28 GHz: the 6 GHz clamp); UMi LOS (hBS 10, hUT 1.5, d2D 200 m, fc 1.5 and 28 GHz: the 2 GHz clamp); SMa NLOS (hBS 35, hUT 1.5 and 4.5, d2D 200 m, fc 3.5 GHz: the ESD height terms); InF NLOS DL (hBS 1.5, hUT 1.5, d2D 30 m, DS with the file default V/S and with `DS_VS` 4 and 50/11 set in a scenpar copy) | Table 7.5-6 Parts 1, 3, 4, Tables 7.5-7, 7.5-8, 7.5-12, typed | 1e-12; DS mean of InF NLOS DL with the file default (4.55 for the hall's 50/11) 3.6e-4 in lg (rounding bound `lg((30·4.55 + 32)/(30·50/11 + 32))` = 3.52e-4, in a comment) |
| EoD offset formula and sign | custom `EoD_offset_*` keys with every term active (`b` > 0 and 0, `hcl` above and below hUT), d2D {5, 50, 500, 2000}; spot checks against the TR: UMa NLOS (6 GHz clamp) and SMa NLOS (hUT above and below 3.5 m) give `-mu_offset,ZOD` of Tables 7.5-7 and 7.5-12; UMa LOS gives 0 | EoD formula typed; TR tables typed | 1e-10 deg |
| values are 10^(mu + sigma x) with x from the correlation root | identity correlation, distinct mu/sigma per LSP, given `rnd`: `lsp = 10^(mu + sigma·s)` (KF, SF, XPR: `10^(0.1·(...))`) | typed | 1e-12 relative |
| the Cholesky root follows the TR order | a custom positive definite matrix with all 28 correlations non-zero, mu = 0, sigma = 1 (dB for KF, SF, XPR): links with `s = e_j` give column j of `Q` built by the test-side Cholesky of the permuted matrix; `x_SF = s_SF`; `Q·Q^T = R` | test-side Cholesky | 1e-12 |
| the symmetric square root is the principal root of R | `cholesky = false`, links with `s = e_j` give the columns of `X`: `X` symmetric, `X·X = R`, `X` positive definite (test-side Cholesky of `X` succeeds); for the custom matrix above, for a matrix with one correlation of 0.999999 (smallest eigenvalue about 1e-6) and for the identity (`X = I`) | uniqueness of the SPD root | 1e-12 absolute for `max abs(X·X - R)`, `max abs(X - X^T)`, `max abs(X - I)`; a comment states the reason: the iteration ends at rounding level (residual about 2e-14 for the 0.999999 matrix) |
| sample covariance of the log values equals the correlation matrix | 20000 links with Box-Muller `rnd`, both roots, `clip_spreads = false`; scenpar: the 28 correlation keys of the UMa NLOS file, the `_mu` keys of the 8 LSPs 0 and the `_sigma` keys 1 (dB for KF, SF, XPR), all other keys at their defaults (no frequency, distance, height, elevation or V/S terms), so that `x = lg(lsp)` for DS, ASD, ASA, ESD, ESA and `x = 10·lg(lsp)` for KF, SF, XPR (the file's own KF std is 0, and clipping would falsify `x`): sample covariance of `x` within 0.05 of `R` | R | 0.05 = 5 standard errors (`sqrt(2/n)` = 0.010 on the diagonal, `sqrt((1 + r²)/n)` <= 0.010 off it) |
| spreads are clipped at 104 and 52 degrees | mu of ASD, ASA 2.5 and of ESD, ESA 2.0: clipped with `clip_spreads = true`, not with false; `lsp_mu` the same in both | — | exact |
| the same random values apply to all frequencies | `(lg DS - mu)/sigma` equal at 3 frequencies | TR 7.6.5 | 1e-12 |
| sf_sigma from path_loss replaces the SF standard deviation | `lsp SF = 10^(0.1·sf_sigma·x_SF)`, `lsp_sigma` slice 2 = `sf_sigma`, other slices unchanged | typed | 1e-12 |
| LSP_fc_min clamps the frequency | with `LSP_fc_min = 6`: `lsp_mu`, `lsp_sigma` at 2 GHz equal those at 6 GHz, differ at 10 GHz | — | exact / > 1e-6 |
| InF delay spread uses V/S | InF LOS file: the file default (4.27 for 47/11) and `DS_VS` set to 4 and 50/11 in a scenpar copy; `DS_VS = 0` gives no V/S term | Table 7.5-6 Part-3 typed | 1e-12 against the formula with the `DS_VS` used; the file default against the TR value with V/S = 47/11: 2.5e-4 in lg (rounding bound `lg((26·47/11 + 14)/(26·4.27 + 14))` = 2.46e-4, in a comment) |
| absolute TOA | `absTOA_mu = -30` gives 0, -29.99 active; value `10^(mu + sigma·rnd(:,8))`; `absTOA_Lmax` bound on both sides of `2L/c` | rule typed (active above -30, bound `2·L/c`, `c = 299792458`) | 1e-12 relative |
| ground permittivity interpolates dry, medium and wet ground | U = 0, 0.25, 0.5, 0.75, 1 at f 1, 10, 28 GHz; imaginary part >= 0 | permittivity model typed | 1e-12 |
| optional outputs alone and together give the same values | every optional output alone vs. all together; pre-sized outputs; `n_link = 0`; one `tx_pos` row vs repeated | — | exact |
| input validation | sections for every check of 3.4.1 (incl. `rnd` 10 columns, NaN in `rnd`, `sf_sigma` size, `sf_sigma` -1 and NaN, `DS_VS > 0` with a = b = 0 and `LSP_fc_min` -1 (both `scenpar_validate` messages), a not positive definite matrix, column 9 = 1.5 only when `gr_eps_re` is requested) | — | — |

### 5.5 `test_los_probability.cpp`

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| TR 38.901 models match Table 7.4.2-1 | each model on both sides of every breakpoint: RMa d {5, 10, 10.001, 500, 5000}; UMi {0, 18, 18.5, 100, 1000}; UMa hU {1.5, 15, 22.5, 25} x d {10, 18, 30, 100, 400, 1000}; InH Mixed {1, 1.2, 3, 6.5, 6.6, 50}; Open {3, 5, 20, 49, 50, 200}; InF SL/DL/SH/DH with defaults and with explicit arguments, hB 8, hU 1.5, d {1, 5, 20, 50}; HH; SMa with hB 35 (Table 7.2-5), r_veg {0, 0.1, 0.2}, hU {1.5, 4.5, 13.5}, d {5, 10, 11, 100, 1000} | typed TR, limited to [0, 1] | 1e-14 |
| 38.811, QuaDRiGa and fixed models | 38.811 tables at elevations 5, 10, 45, 87, 90 deg and below the horizon; Industrial with hB 4, 8, 12; LOS, NLOS | typed (elevation tables, QuaDRiGa formula) | 1e-14 |
| probabilities stay in [0, 1] | UMa at d 18.1 m with hU 20 m (TR formula 1.00246) and hU 22.5 m (1.00470) gives 1; at d 20 m, hU 20 m the TR value 0.97779 (below 1, not limited); every model over a grid of d and heights in [0, 1] | limit to [0, 1]; typed TR | exact; 1e-14 |
| InF edge rules | hU >= h_c gives 1; hB < hU < h_c and hB = hU < h_c give the SL rule; r = 0 gives 1 | edge rules typed | 1e-14 |
| SMa edge rules | hB 35 m (Table 7.2-5) unless stated: hB <= hU gives 1; hU 18 m (above the vegetation height of 15 m) sets the vegetation term to 0 | edge rules typed | 1e-14 |
| soft LOS is 0.5 at G = -F(d) and matches p over many G | `rnd_g = -F(p)` with F from the test-side probit gives 0.5 (bisection for `x <= 0` on `0.5·erfc(-x/sqrt(2)) = min(p, 1 - p)` until the interval is below `1e-15·max(1, abs(x))`, `F = x` for `p <= 0.5` and `-x` otherwise; the symmetric form keeps the upper tail exact, where `0.5·erfc(-x/sqrt(2))` would round to `p`); quantile grid `G_k = F((k + 0.5)/K)`, K = 10000: fraction of `los_soft > 0.5` equals p within 1/K; p = 1 gives 1, p = 0 gives 0; eq. (7.6-18) typed at fc 3.5 and 28 GHz | test-side probit, eq. (7.6-18) | `k·1e-10·max(1, abs(F))/pi + 1e-15` (from the probit target, `k = sqrt(20/lambda)`); counts within 1/K |
| the inverse normal CDF of the soft LOS is accurate | `38.901_RMa` with `d = 10 - 1000·ln(p_target)` for p on a logarithmic grid in [1e-12, 0.5] and `d` just above 10 m for p up to 1 - 1e-12; `rnd_g = 0`, fc = 1 kHz (`k = sqrt(20/lambda)` about 8e-3, so `atan` is nearly linear): `F_rec = tan(pi·(los_soft - 0.5)) / k` equals the test-side probit (symmetric bisection, above) of the returned `p_los`, on both tails | symmetric bisection on `std::erfc` | `1e-10·max(1, abs(F))` |
| optional inputs | `los_soft` without `rnd_g` throws; `rnd_g` without `los_soft` is ignored; scalar `h_bs` equals the repeated vector | — | exact |
| input validation | sections for every check of 3.5.1 | — | — |

### 5.6 `test_o2i_loss.cpp`

| Test case | Asserts | Oracle | Tolerance |
|---|---|---|---|
| wall and indoor losses follow Tables 7.4.3-1 and 7.4.3-2 | one-hot weights for `low`, `high`, `low_A`; fixed `rnd` per UT; f 3.5, 28, 60 GHz: `loss = PL_tw + 0.5·d2din + sigma_P·N` | typed tables | 1e-12 dB |
| the backward compatible model of Table 7.4.3-3 | `legacy_6GHz`: `20 + 0.5·U1·d2d_in_max`, independent of columns 2 and 3 | typed | 1e-12 |
| car penetration loss | `car`: `9 + 5·N`, `car_metal`: `20 + 5·N`, `d2d_in = 0`, independent of f | 7.4.3.2 typed | 1e-12 |
| the model is chosen by the cumulative weights | 1000 UTs with `rnd(:,0) = (k + 0.5)/1000`, weights {0.2, 0, 0.5, 0.3}: counts 200, 0, 500, 300; `u = 1` picks the last non-zero weight; `u = 0` never picks a zero weight | cumulative-weight rule | exact |
| indoor distance is the minimum of two uniforms | `d2d_in = min(U1, U2)·d_max`; grid 100 x 100 of (U1, U2): fraction with `d2d_in <= x·d_max` equals `1 - (1 - x)²` within 0.02 | TR 7.4.3.1 | 1e-12; grid 2/100 |
| optional outputs and sizes | each optional output alone vs together; pre-sized; `n_ut = 0` | — | exact |
| input validation | sections for every check of 3.6.1 (incl. `rnd` with 6 columns, an unknown model name such as `high_v14`) | — | — |

### 5.7 Internal helpers

No test file of its own: the internal header holds only helpers with public users, and each is covered
there with an independent oracle.

| Helper | Covered by |
|---|---|
| `qd_gbsm_lsf_generators`, `qd_gbsm_freq`, `qd_gbsm_positions`, `qd_gbsm_n_coeff` (file-local) | 5.1, 5.2 (`ref_sos` with each generator's own distance, 2D/3D, Tx broadcast, unknown preset) |
| `qd_gbsm_check_links`, `qd_gbsm_check_fc` | the validation cases of 5.1 to 5.6 (messages of 3.0.2) |
| `qd_gbsm_lsp_stats` | 5.4 (formula with every term, frequency-term check), 5.3 (default `sf_sigma`, SF frequency-term check) |
| `qd_gbsm_chol8` | 5.4 (TR-order root against a test-side Cholesky, not positive definite), existing conf tests (positive-definite flag of `qd_scenpar_xcorr`) |
| `qd_gbsm_floor_offset` (file-local) | 5.2 (typed offset rule, zero correlation within a block) |
| `qd_gbsm_eval` (file-local) | 5.2 (`ref_sos`, two-device formulas, floors in single and dual mobility, uniform map) |
| `qd_gbsm_remove_bias` (file-local) | 5.1 (zero drop mean with `ref_sos`, single and dual mobility, 2D/3D) |
| symmetric square root (file-local) | 5.4 (`X·X = R`, symmetry, positive definite) |
| probit (file-local) | 5.5 (accuracy through `los_soft`) |

### 5.8 Changes to existing tests

- Run 1: `test_conf_file_read.cpp`, `test_scenpar_validate.cpp`, `test_conf_file_write.cpp` (165 -> 171 keys, 27 ->
  33 new keys, 166 -> 172 assignments with `SubpathMethod`, the key-order oracle extended by the 6 names, the new
  cross-key rule of 4.2 with a negative case in the validation test): the 6 keys are added to the
  `QD_NONNEG` lists of the cases "value checks per validation class" in `test_conf_file_read.cpp` and
  `test_scenpar_validate.cpp`; the cross-key rule through `scenpar_validate`, through `conf_file_read` (message
  with file name and the line of `DS_VS`) and through `conf_file_write`; in `test_conf_file_write.cpp` a case "the
  new keys land in their section and paragraph and are not written at their default" (`LOS_state_lambda`,
  `IO_state_lambda` in the paragraph after `SC_lambda` of the small-scale section, `O2I_d2D_in_max` in its own
  paragraph of "Model parameters", the `DS_VS` keys in the DS paragraph; none of the 6 lines at value 0; at
  non-default values the complete written line of each key is compared, with its comment text of 4.1) and a
  case "the DS overview line shows the V/S term" next to the ESD overview case (`DS_VS > 0` gives
  `% DS  = log10( 26 * V/S + 14 ) - 9.35 + Xds * ( 0.15 )`, `DS_VS = 0` the line of today); `test_conf_files_v19.cpp`: include
  `quadriga_gbsm.hpp`, the InF DS checks (`DS_mu` -9.35 / -9.44, `DS_VS`, `DS_VS_a`, `DS_VS_b` and
  `DS_mu + lg(a·VS + b)` against the TR formula for the hall of 7.8.4; `DS_VS` equals the hall's V/S rounded to 2
  decimals, the mean is within `lg(1.001)` = 4.3e-4 of the TR formula, the 0.1 % of the precision rule of 1.2),
  the values of the 6 keys per file (4.4.1),
  `LOS_state_lambda` and `IO_state_lambda` 0 in all other files; `absTOA_Lmax` 120 (InF_NLOS_SL, _DH,
  Indoor_NLOS) and 300 (InF_NLOS_DL, _SH) against L = max(length, width, height) of the halls typed from TR 7.8.4
  and Table 7.2-2, 0 in all other files (the existing check `absTOA_Lmax == 0` for every file is changed); `test_conf_file_list.cpp`: include. MEX tests
  `test_conf_file_read.m`, `test_scenpar_validate.m`, `test_conf_file_write.m` (166 -> 172 fields, field indices of
  the new keys; the `DS_VS` cross-key error through `scenpar_validate`); Python tests of the four functions call
  `quadriga_lib.gbsm.*`, counts 166 -> 172, the `DS_VS` cross-key error (`ValueError`), plus one check that
  `quadriga_lib.tools` has no `conf_file_read` any more. Comments with planning labels in the Catch2 conf tests
  that Run 1 edits are reworded to state the rule (9.1): "decision O10" in `test_conf_files_v19.cpp`, "Decision Q1",
  "Q2", "Q3" in `test_conf_file_read.cpp` and `test_conf_file_list.cpp`, "Decision B1" and "review point A" in
  `test_conf_file_write.cpp`.
- Run 3a: `test_conf_files_v19.cpp` case "path-loss constants of RMa and SMa follow Table 7.4.1-1": each file value
  equals the exact constant rounded as in 4.4.2 (RMa `A1`, `B`, `Bn`, `E` to 2 decimals, `D` to 6; SMa `A1` to 3,
  `B` and `E` to 2), tolerance half a unit of the last written digit plus 1e-12; UMa, UMi, InH, InF `E` equals 13.33
  (40/3 to 2 decimals); SMa NLOS asserts `LOSmax == 0` and the absence of `A1, A2, B, C, E, hE`; new checks of
  `PL_hE_rand` (UMa files) and `PL_Ax/Bx/Cx` (InF-DL);
  the comment "E and hE ... needed by the QuaDRiGa model" is removed. `test_conf_file_write.cpp` case "path-loss
  comments and formulas depend on the model": one check per new `PL_` key comment (`hE_rand`, `LOSmax`, `Ax`,
  `Bx`, `Cx`).


## 6. Wrappers

General rules (both APIs): section "Geometry-based stochastic model"; one-line summaries identical to the C++ blocks;
thin wrappers, no numerical logic; `scenpar` is the struct / dict returned by `conf_file_read` or
`scenpar_validate` (the entry `SubpathMethod` is accepted and ignored), `[]` / `None` = all defaults; `plpar` is the
struct / dict of `conf_file_read` whose entry `model` is passed as `pl_model` (no `model` entry or `[]` / `None` =
empty model, zero loss). Helpers: MEX `qd_mex_struct2scenpar`, `qd_mex_struct2plpar`
(`src/mex_quadriga_lib_functions.hpp`, include `mex_quadriga_lib_functions.hpp`); Python `qd_python_dict2scenpar`,
`qd_python_dict2plpar` (`api_python/python_quadriga_adapter.hpp`). `floor_id` is read as double, checked to be
non-negative integers below 2^53 (`Entries in 'floor_id' must be non-negative integers below 2^53.`; from 2^53 on
a double does not hold every integer, so larger IDs would change silently) and converted to `arma::uvec`. The
documentation blocks of `lsf_draw` state that in dual mobility `floor_id` is the floor of both devices (3.2).

### 6.1 MEX (`api_mex/<name>.cpp`, `tests/mex_tests/test_<name>.m`)

| Function | Usage | Inputs (defaults) | Notes |
|---|---|---|---|
| lsf_init_phases | `phase = quadriga_lib.lsf_init_phases( scenpar, seed, tx_pos, rx_pos, dual_mobility, remove_bias, preset, horizontal );` | scenpar; seed -1 (int64); tx_pos `[]`; rx_pos `[]`; dual_mobility false; remove_bias true; preset 'Comb256'; horizontal true | nrhs 1..8, nlhs <= 1; `phase` `[n_coeff, 11, 2]` copied after the call (size from the preset) |
| lsf_draw | `rnd = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, preset, horizontal );` | floor_id `[]` (all on floor 0); rest as above | nrhs 4..8, nlhs <= 1; `rnd` `[n_link, 11]` via `qd_mex_init_output`; floor IDs are labels, no index shift |
| path_loss | `[ pl, sf_sigma ] = quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d2d_in, rnd_he );` | d2d_in `[]` (= geometric distance, TR); rnd_he `[]` | nrhs 5..7, nlhs <= 2; outputs `[n_link, n_freq]` pre-allocated |
| lsp_generate | `[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = quadriga_lib.lsp_generate( scenpar, fc, tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads );` | sf_sigma `[]`; cholesky true; clip_spreads true | nrhs 5..8, nlhs <= 7; the requested outputs (1 to nlhs) pre-allocated (`[n_link, n_freq, 8]`, `[n_link, 1]`, `[n_link, n_freq]`), the C++ outputs beyond nlhs passed as `nullptr`; so column 10 of `rnd` (MATLAB index) is checked only when `gr_eps_re` or `gr_eps_im` is requested, which the documentation block states |
| los_probability | `[ p_los, los_soft ] = quadriga_lib.los_probability( model, d2d_out, h_bs, h_ut, d_clutter, r_clutter, h_clutter, rnd_g, fc );` | d_clutter, r_clutter, h_clutter NaN (= model default); rnd_g `[]`; fc `[]` | nrhs 4..9, nlhs <= 2; `los_soft` is `[]` when `rnd_g` or `fc` is empty, otherwise `[n_link, n_freq]` |
| o2i_loss | `[ loss, d2d_in, model_index ] = quadriga_lib.o2i_loss( fc, rnd, models, weights, d2d_in_max );` | d2d_in_max 25 | nrhs 4..5, nlhs <= 3; `models` cell array of char (or one char array); `model_index` uint64, **1-based** (shift +1 after the call) |

Optional C++ outputs whose MEX output is not requested (beyond nlhs) are passed as `nullptr` in every wrapper.
Numeric inputs, `floor_id` included, accept the types that the MEX helpers read (double, single, int32, uint32,
int64, uint64); other integer types such as `uint8` are rejected by the helpers (user decision after Run 4a). Zero
links: the MEX helpers turn an empty `[0, 3]` position array into 0x0, so the position-based wrappers reject zero
links with the C++ message on the column count; this restriction is accepted and the MEX documentation blocks do not
state the `n_link = 0` rule of the C++ blocks (C++ and the Python wrappers support zero links).
Documentation blocks follow SPEC_MATLAB_API (UPPERCASE heading, `## Usage:`, 100-character lines). Each MEX test
covers the checklist of SPEC_MATLAB_API "What every test covers" with these specific points:

- lsf_init_phases: size `[256, 11, 2]`; same seed equal, `seed = []` default runs; `int64(7)` seed equals 7;
  positions as `single`; mirror relation of slice 2 (MATLAB index) without positions; bias removal changes slice 1
  for 3 positions or more; errors: argument counts, unknown preset (C++ message `Comb999`), dual mobility without
  `tx_pos` (C++).
- lsf_draw: `[n_link, 11]` double; z-independence in 2D, z-dependence with `horizontal = false`; `floor_id = []`
  equals all zeros; `floor_id` with `-1`, `1.5` and `2^53` raise the wrapper error; `uint32` and `int64` floor IDs accepted; dual
  mobility swap of columns 4/5 (MATLAB 1-based) with swapped positions; `phase` of the wrong size raises the C++
  error.
- path_loss: logdist via a hand-made `plpar` struct (`model = 'logdist'`, `A = 20`, `B = 30`, `C = 20`) at d3D =
  100 m, 1 GHz gives 70 dB; `plpar = []` gives zeros; a v19 file read with `conf_file_read` runs; `sf_sigma`
  output with `SF` field; `d2d_in` given changes the result (legacy distance); `rnd_he` with the UMa
  file; error: unknown model (C++).
- lsp_generate: sizes of all 7 outputs; `rnd = zeros` gives `lsp = 10.^lsp_mu` for DS; `clip_spreads = false`
  allows ASD > 104; `cholesky = false` changes the KF/SF mixing; `sf_sigma` replaces slice 3 (MATLAB index) of
  `lsp_sigma`; `DS_VS` in the struct changes the InF DS; column 10 of `rnd` = 1.5 runs with `nlhs = 1` and raises
  the C++ error with `nlhs = 3`; error: `rnd` with 10 columns (C++).
- los_probability: `38.901_UMi` at d = 10 m gives 1; `'NLOS'` gives 0; `h_bs` scalar vs vector; `los_soft` `[]`
  without `rnd_g`; `los_soft` `[n_link, n_freq]` with both; `d_clutter = []` equals NaN equals omitted; error:
  unknown model (C++).
- o2i_loss: `models = 'car'` (char) and `{'car'}` equal; `model_index` uint64 starting at 1; `rnd` with 6 columns
  (C++ error); weights not summing to 1 (C++ error).

### 6.2 Python (`api_python/qpy_<name>.cpp`, submodule `gbsm`, `tests/python_tests/test_<name>.py`)

Dispatcher `api_python/python_gbsm.cpp` (created in Run 1 with the four conf functions; Run 4b adds the six), registered in
`python_main.cpp` as `py::module_ gbsm = m.def_submodule("gbsm", "Geometry-based stochastic model");`; the file is
added to the source list in `CMakeLists.txt` (line ~639, after `python_tools.cpp`). Mandatory arrays use
`.none(false)` as in `python_sos.cpp`.

```
phase = quadriga_lib.gbsm.lsf_init_phases( scenpar, seed=-1, tx_pos=None, rx_pos=None, dual_mobility=False,
    remove_bias=True, preset="Comb256", horizontal=True )
rnd = quadriga_lib.gbsm.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id=None, dual_mobility=False,
    preset="Comb256", horizontal=True )
pl, sf_sigma = quadriga_lib.gbsm.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d2d_in=None, rnd_he=None )
# complex=True (default)
lsp, abs_toa, gr_eps, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( scenpar, fc, tx_pos,
    rx_pos, rnd, sf_sigma=None, cholesky=True, clip_spreads=True, complex=True )
# complex=False
lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma = quadriga_lib.gbsm.lsp_generate( ...,
    complex=False )
p_los, los_soft = quadriga_lib.gbsm.los_probability( model, d2d_out, h_bs, h_ut, d_clutter=None, r_clutter=None,
    h_clutter=None, rnd_g=None, fc=None )
loss, d2d_in, model_index = quadriga_lib.gbsm.o2i_loss( fc, rnd, models, weights, d2d_in_max=25.0 )
```

- `scenpar`, `plpar`: `py::handle` (dict or None). `seed`: `long long`. Flags (`dual_mobility`, `remove_bias`,
  `horizontal`, `cholesky`, `clip_spreads`, `complex`): `py::handle`, read with `qd_python_flag(handle, default,
  name)` as in the four conf functions of the submodule (SPEC_PYTHON_API "Flags"): `None` selects the default, bool
  and `numpy.bool_` give their value, any other type (float, int, str) raises `TypeError` (`Input '<name>' must be
  a bool or None.`); the `m.def` defaults stay `true` / `false`. `d_clutter`, `r_clutter`, `h_clutter`:
  `py::handle`, `None` (the default) = NaN = the TR default of the model, else a real scalar (`ValueError`
  `Input '<name>' must be a real scalar or None.`), as MATLAB accepts `[]`.
- Outputs: `phase` `(n_coeff, 11, 2)` copied after the call; all others pre-allocated with
  `qd_python_init_output`; `model_index` uint64 (pre-allocated, C++ type kept, 0-based); `abs_toa` 1D `(n_link,)`;
  `gr_eps` complex128 `(n_link, n_freq)` built with `qd_python_copy2numpy<double, std::complex<double>>`; `los_soft`
  is `None` when `rnd_g` or `fc` is None.
- `models`: `qd_python_list2vector_Strings` (str or list of str). `floor_id`: float array, checked, converted.
- `lsp_generate` always returns `gr_eps` (or `gr_eps_re`, `gr_eps_im`), so the C++ check of column 9 of `rnd`
  always applies: its documentation block states that column 9 must be in [0, 1].
- Each Python test covers the checklist of SPEC_PYTHON_API "What every test covers"; the specific points of 6.1
  apply with 0-based indices (`model_index` starts at 0) and these additions: keyword call with all Usage names;
  C-ordered, Fortran-ordered and strided `rx_pos` give the same result; `float32` positions accepted; `complex`
  flag of `lsp_generate` (tuple length 6 vs 7, `gr_eps.real == gr_eps_re`); `quadriga_lib.gbsm.conf_file_read` of a
  v19 file feeds `path_loss` and `lsp_generate` directly; `TypeError` for a missing mandatory argument and an
  unknown keyword; `ValueError` from C++ for an unknown path-loss model; every flag: `None` equals the default and
  a non-bool value (e.g. 2.5) raises `TypeError`; clutter arguments `None` equal NaN.


## 7. Implementation runs

Seven sequential runs, each one agent session: Run 1, Run 2a, Run 2b, Run 3a, Run 3b, Run 4a, Run 4b. A run starts only after the previous
run's report was accepted and the maintainer committed. Every run reads the sections of this spec named in its "Read"
line (always including 1, 9 and the parts of 2.1 it adds), `md/SPEC_CPP_API.md` and, for wrappers, the wrapper SPEC
file of its API. The constraints of section 9 apply to every run. Each run adds to `include/quadriga_gbsm.hpp` only
the declarations it defines (2.1).

Every run writes `md/reports/r<N>_report.md` (N = 1, 2a, 2b, 3a, 3b, 4a, 4b) with the sections: **Summary** (what was
done, 5 lines); **Files** (created / changed); **Decisions and deviations** (every choice made within the spec,
every deviation from the spec or a SPEC file, with the spec section; the SPEC review points A/B/C are answered here
instead of stopping); **Test results** (commands and the actual summary lines of the runs: Catch2 test-case and
assertion counts, MEX suite result per MATLAB/Octave version, pytest summary, the objdump gate output where
required, load average before heavy runs); **Warnings** (compiler warnings, none expected); **Open points**
(proposals for the spec, things not tested); **Suggested commit cut** (themes and file lists; the agent does not
commit).

### Run 1 — header move, new keys, Python submodule

Read: 1, 2.1, 3.7, 4.1, 4.2, 4.4.1, 5.8 (Run 1 part), 6 (general rules), 9.

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
5. Update the Catch2 tests (5.8, Run 1 part, including the rewording of the planning labels), run the conf tests,
   then the full suite.
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

### Run 2a — internal header, `lsf_init_phases`, `lsf_draw`

Read: 1, 2 (2.1 declarations of the two functions, 2.2), 3.0, 3.1, 3.2, 5.0 to 5.2, 5.7, 8, 9; source files
`include/quadriga_sos.hpp`, `src/sos_val.cpp` (device modes, kernel selection), `src/sos_remove_bias.cpp`,
`src/sos_preset.cpp` (interface only), the SOS helpers in `src/quadriga_lib_helper_functions.hpp`.

Files: `src/quadriga_gbsm_helpers.hpp` (new: documentation block, `qd_gbsm_gen`, `qd_gbsm_lsf_generators`,
`qd_gbsm_freq`, `qd_gbsm_positions`), `src/lsf_init_phases.cpp`, `src/lsf_draw.cpp` (new),
`include/quadriga_gbsm.hpp` (two declarations), `tests/catch2_tests/test_lsf_init_phases.cpp`, `test_lsf_draw.cpp`
(new), `tests/catch2_tests/qd_test_diag.hpp` (area line `QD_DIAG_GBSM`).

Steps (SPEC_CPP_API phases, without stops): declarations and stubs (section statement, first documentation block,
outputs sized and zeroed), build; the helper header; the tests of 5.1 and 5.2; run against the stubs and record in
the report which checks fail for the right reason; implement `lsf_init_phases`, then `lsf_draw` (the generic
phase-set functions `qd_gbsm_remove_bias`, `qd_gbsm_eval`, `qd_gbsm_floor_offset` with the signatures of 3.0.5);
iterate until green;
final documentation blocks (3.1, 3.2); full suite.

Acceptance: all new cases pass; `make test` passes; no warnings; documentation blocks complete. No MEX or Python
build needed.

### Run 2b — `lsp_generate`, shared LSP mean/std and Cholesky

Read: 1, 2 (2.1 declaration), 3.0, 3.4, 5.0, 5.4, 5.7, 8, 9; `src/quadriga_gbsm_helpers.hpp` and the
documentation blocks written in Run 2a; `qd_scenpar_xcorr` in `src/qd_scenpar.cpp`.

Files: `src/quadriga_gbsm_helpers.hpp` (add `qd_gbsm_lsp_mu_sigma`, `qd_gbsm_chol8`), `src/lsp_generate.cpp` (new;
the symmetric square root file-local), `src/qd_scenpar.cpp` (`qd_scenpar_xcorr` calls `qd_gbsm_chol8`),
`include/quadriga_gbsm.hpp` (one declaration), `tests/catch2_tests/test_lsp_generate.cpp` (new).

Steps: declaration and stub, build; the helpers; the tests of 5.4; run against the stub; implement; refactor
`qd_scenpar_xcorr` and run the conf tests; documentation block (3.4.4); full suite.

Acceptance: all new cases pass; the existing conf tests pass unchanged; `make test` passes; no warnings.

### Run 3a — `path_loss`, path-loss keys and constants

Read: 1, 2 (2.1 declaration), 3.0, 3.3, 4.3, 4.4.2, 5.0, 5.3, 5.8 (Run 3a part), 9, 11.1; `qd_conf_pl_comments` in
`src/qd_scenpar.cpp`; `src/quadriga_gbsm_helpers.hpp`.

Files: `src/path_loss.cpp` (new), `include/quadriga_gbsm.hpp` (one declaration), `src/qd_scenpar.cpp`
(`qd_conf_pl_comments` rows), the v19 files of 4.4.2, `tests/catch2_tests/test_path_loss.cpp` (new),
`test_conf_files_v19.cpp`, `test_conf_file_write.cpp` (comment rows).

Steps: declaration and stub, build; the tests of 5.3; run against the stub; the file edits of 4.4.2 before the TR
test is run against the implementation (it needs the rounded constants of 4.4.2 and the new keys);
implement; the writer comment rows and their test; the `test_conf_files_v19.cpp` updates; documentation block
(3.3.7); full suite.

Acceptance: all new cases pass; `make test` passes; no warnings.

### Run 3b — `los_probability`, `o2i_loss`

Read: 1, 2 (2.1 declarations, 2.3), 3.0, 3.5, 3.6, 5.0, 5.5, 5.6, 9, 11.2.

Files: `src/los_probability.cpp` (new; the probit file-local), `src/o2i_loss.cpp` (new), `include/quadriga_gbsm.hpp`
(two declarations), `tests/catch2_tests/test_los_probability.cpp`, `test_o2i_loss.cpp` (new).

Steps: declarations and stubs, build; the tests of 5.5 and 5.6; run against the stubs; implement; documentation
blocks (3.5.5, 3.6.3); full suite.

Acceptance: all new cases pass; `make test` passes; no warnings.

### Run 4a — MEX wrappers, documentation sweep

Read: 1, 2, 3 (documentation-block items), 6 (general rules, 6.1), 7, 9, `md/SPEC_MATLAB_API.md`, and the C++
documentation blocks written in Run 2a to Run 3b.

Files: `api_mex/lsf_init_phases.cpp`, `lsf_draw.cpp`, `path_loss.cpp`, `lsp_generate.cpp`, `los_probability.cpp`,
`o2i_loss.cpp` (new); `tests/mex_tests/test_<name>.m` (6 new); `SECTION_DESC` blocks in `src/qd_scenpar.cpp` and
`api_mex/conf_file_read.cpp`; `See also` lines of the ten C++ blocks (each new function names its
neighbours in the data flow of 1.1).

The `SECTION_DESC` (C++ version; MEX and Python adapt the names) explains in 10 to 20 lines: what a geometry-based
stochastic model is (TR 38.901 clause 7.5 steps 1 to 4 here), the conf files (`data/config`, 23 v19 files), the
data flow of 1.1 with the function names as `[[links]]` (incl. the layout order `o2i_loss` before
`los_probability`), the generator index of 2.2, the units, and that the LOS state and O2I draws are made by the
caller with [[sos_rand]] / [[sos_randn]] at the distances of the keys `LOS_state_lambda`, `IO_state_lambda`.

Steps: MEX wrappers and tests (SPEC_MATLAB_API order: wrapper, then test); `SECTION_DESC` (C++ and MEX); `See
also` lines; full gates.

Acceptance: MEX suite passes in Octave and MATLAB R2024a (R2026a if available), objdump gate clean; `make test`
passes; no warnings in any build.

### Run 4b — Python wrappers, release notes

Read: 1, 2, 6 (general rules, 6.2), 7, 9, `md/SPEC_PYTHON_API.md`, the C++ and MEX documentation blocks.

Files: `api_python/qpy_<name>.cpp` (6 new), `api_python/python_gbsm.cpp` (registrations),
`tests/python_tests/test_<name>.py` (6 new), `SECTION_DESC` in `api_python/qpy_conf_file_read.cpp`.

Steps: Python wrappers, registrations, tests; `SECTION_DESC` (Python); release-note text (new functions, header
move, Python submodule move `tools` -> `gbsm` (breaking), new keys, changed v19 files incl. the path-loss changes of
up to 0.014 dB and the TOA bound of the InF and InH NLOS files, the reduced O2I model set, the bias-removal change
against QuaDRiGa, the dual-mobility ASD/ASA and ESD/ESA values without QuaDRiGa's extra coupling, the LSP standard
deviation limited at 0) in the report
(`README.md` has no release-notes section; the maintainer decides where the text goes); full gates.

Acceptance: `make python` and `make python_test` pass; `make test` passes; no warnings in any build.


## 8. Requirements from the later SSF port

Read: `gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `get_channels.m` (lines 380 to 480), `add_sdc.m` and TR 7.5 steps 5
to 12, 7.6.2, 7.6.3.1, 7.6.13 to 7.6.16. The small-scale step needs from the LSF module:

| Quantity needed by the SSF step | Where (MATLAB) | Provided by | Status |
|---|---|---|---|
| Per-link, per-frequency DS, ASD, ASA, ESD, ESA (linear, deg) | `gen_ssf_parameters.m` lines 243 to 248 (`h_builder.ds` ...) | `lsp_generate` `lsp` slices 0, 3 to 6 | provided |
| KF per link and frequency (linear; TR steps 5 to 7 use K in dB) | line 248; TR (7.5-3), (7.5-8), (7.5-15) | `lsp` slice 1; dB = `10·lg` | provided |
| SF per link and frequency (step 12) | `get_channels.m` 428 to 480 | `lsp` slice 2 with `sf_sigma` from `path_loss` | provided |
| XPR mean per link and its std per frequency (step 9; QuaDRiGa uses the per-link XPR as mean) | lines 1048 to 1066 (`lsp_vals(8,2)`, `10·log10(xpr)`) | QuaDRiGa default mode: mean `10·lg` of `lsp` slice 7 per link, per-ray std `0.1 · lsp_sigma` slice 7 per frequency (the factor 0.1 is QuaDRiGa's: `xpr_sigma = 0.1 · lsp_vals(8,2,f)`); QuaDRiGa baseline mode and TR variant (mu_XPR, sigma_XPR per ray, eq. 7.5-21): `lsp_mu` and `lsp_sigma` slice 7 (QuaDRiGa's baseline takes the first frequency) | provided |
| LSP means without randomness (`lsp_vals`) | `add_sdc.m`, `gen_ssf_parameters.m` | `lsp_mu`, `lsp_sigma` (call with `rnd = zeros`) | provided (3.4.4), not 1:1: `lsp_vals` is the per-builder coefficient table without distance, height and elevation terms (e.g. `add_sdc` uses `mean(lsp_vals(5,1,:))`), while `lsp_mu` is per link with all terms. No geometry reproduces `lsp_vals` for files with elevation terms (`lg d`, `lg h` and `lg(atan(h/d))` cannot vanish together; d = h = 1 m gives `lg(pi/4) = -0.105`). The SSF port and NG therefore use the per-link `lsp_mu` / `lsp_sigma`, or their mean over the builder's links where a per-builder value is needed |
| Per-link ZSD mean for the cluster ESD `c_ZSD = 3/8 · 10^mu_lgZSD` (7.6-7, key `PerClusterES_D_scale`) | not yet in NG | `lsp_mu` slice 5 (after the ESD terms and the floor) | provided |
| EoD offset (step 7, eq. 7.5-19 `+ mu_offset,ZOD`) | not in NG today | `eod_offset_deg` (elevation, positive up) | provided |
| Absolute TOA (7.6.9, NLOS only) | lines 813 to 818 | `abs_toa` | provided |
| Ground permittivity per link and frequency (7.6.8 extension) | lines 112, 262 to 265 | `gr_eps_re`, `gr_eps_im` | provided |
| Soft LOS state (7.6-19 combination) | — | `los_probability` `los_soft` | provided; the combination of H_LOS and H_NLOS is SSF and needs per link a LOS and an NLOS builder (two scenpar sets) with independent LSF phase sets, i.e. two `lsf_init_phases` calls with different seeds |
| Cluster-number range D1, D2 (7.6.15) | — | keys `NumClusters_min`, `NumClusters` | provided; the per-link draw is an SSF generator |
| Per-frequency handling (7.6.5: same x, independent per-cluster shadowing) | `xpr_sos`, `pin_sos` per frequency | LSF: one `x` for all frequencies; SSF: own generators | provided for LSF |
| Anchor-frequency LSPs of the alternative multi-frequency method (7.6.5.1: DS0, ASD0, ASA0, ZSD0, ZSA0 at e.g. 2 GHz) | — | `lsp_generate` with the anchor frequency added to `fc` (same `x`, so the anchor and the frequencies of interest are consistent); `clip_spreads` applies to the anchor as well, since 7.6.5.1 takes the spreads "according to Step 4", which includes the clipping | provided |
| Path loss for step 12 | `get_pl` | `path_loss` | provided |
| Spatially consistent cluster and ray variables (7.6.3.1: delays, shadowing, offsets, signs, coupling, XPR, phases; SC_lambda; same floor rule; mirrored for dual mobility) | `init_sos.m` lines 168 to 237 (`path_sos`, `xpr_sos`, `pin_sos`, `clst_dl_sos`) | **not** the LSF phase set (fixed 11 generators, `[n_coeff, 11, 2]`); the SSF port defines its own phase sets and evaluates them with `qd_gbsm_eval`, `qd_gbsm_floor_offset` and `qd_gbsm_remove_bias`, which Run 2a writes generically over the generator table (`qd_gbsm_gen`: lambda, distribution, mirror partner, bias flag) with the signatures of 3.0.5; the SSF functions that create and evaluate these sets are added to `src/qd_gbsm_phases.cpp`, so the machinery stays file-local there | requirement on Run 2a |
| Floor separation of cluster variables (7.6.3.3 last paragraph) | — | the floor offsets of 3.2.1 inside the generic evaluation, with the same per-UT floor IDs as `lsf_draw` (derived by the layout, 10.3) | requirement on Run 2a |
| Seeds of the SSF phase sets, independent of the LSF set | `init_sos.m` (one seed per `qd_sos` object) | `sos_init_phases` fills column g independently of the number of generators, so an SSF set drawn with the builder's LSF seed would repeat the LSF phases in its first 11 columns and, at an equal decorrelation distance (e.g. `SC_lambda` 50 m and an LSF lambda of 50 m in UMa NLOS), reproduce the LSF fields. Rule: the SSF sets take the columns from 11 on of one `sos_init_phases(n_coeff, 11 + n_ssf, seed)` call with the builder's seed (columns 0 to 10 are the raw LSF phases), or use a seed derived from the builder seed by a fixed rule. The first form requires the `n_coeff` of the LSF set: the matrix is filled column by column from one random stream, so with another `n_coeff` the SSF columns would reuse parts of the stream of the LSF columns; SSF sets with another `n_coeff` use a derived seed. The builder therefore stores an explicit seed (drawn once when the caller passes -1) or the SSF phase cubes | requirement on the SSF port and NG |
| Grouping of the evaluation by decorrelation distance | `xpr_sos`: clusters · (M + 1) · n_freq generators at one `SC_lambda` | `qd_gbsm_freq` builds one frequency table per distinct lambda with a generator-to-table index; the evaluation and the bias removal make one call per table (3.0.5) | requirement on Run 2a |
| Dual-mobility flag and the Tx/Rx-side conventions | `h_builder.dual_mobility` | `dual_mobility` argument, slice convention of 2.2 | provided |
| O2I flag (step 7: mean ZOA 90 deg for O2I), LOS/NLOS state | scenario of the builder | builder level, not LSF | n/a |
| d3D for the propagation delay `d3D/c` (7.6-43) | geometry | caller | n/a |
| Per-UT `Pr_sns` (7.6.14), near-field and polarization-variability draws | — | SSF generators | n/a |

Answer to "must the LSF phase set be extensible for SSF generators": no. It stays fixed at 11 generators (contract
of 2.1); the SSF port gets its own `ssf_init_phases`-type function. What must be extensible is the internal
evaluation, which is why Run 2a writes the evaluation and the bias removal generic over the generator table (3.0.5).
No public API was added for SSF readiness.


## 9. Constraints (into every run)

### 9.1 Code

- `double` only; C++17; no warnings with `-Wall -Wextra -Wpedantic`; must compile with MSVC (no GCC extensions, no
  VLAs); OpenMP 2.0 only (signed loop variables, `OMP_SAFE_CALL`), results independent of the number of threads.
  OpenMP is optional in the new code (only loops over links in `path_loss`, `lsp_generate`, `o2i_loss` qualify).
- No BLAS/LAPACK: no `arma::chol`, `eig_sym`, `sqrtmat`, `norm`, `dot`, `solve`, `inv`, `det`, `svd`; the 8x8 Cholesky
  (`qd_gbsm_chol8`, shared with `qd_scenpar_xcorr`) and the symmetric square root are written by hand (3.4.3).
  Armadillo element access and products are fine.
- Precision: functions are double; `lsf_draw` calls `sos_val<double>` with `use_kernel = 0` (AVX2 kernel with
  single-precision sin/cos where available); tests of `lsf_draw` values use the tolerances of 5.0.
- Lean code: reuse existing helpers (`qd_sos_rng`, `qd_sos_uniform`, `qd_interp_linear`, `qd_sos_preset_lookup`,
  `scenpar_validate`); a helper used once is a lambda, a helper used several times within one file is
  `static inline` or in an anonymous namespace, `src/quadriga_gbsm_helpers.hpp` only holds helpers used across
  files (3.0.5). Exception: the generic phase-set functions `qd_gbsm_floor_offset`, `qd_gbsm_eval`,
  `qd_gbsm_remove_bias` are anonymous-namespace functions with the signatures of 3.0.5 in
  `src/qd_gbsm_phases.cpp`, where the SSF port adds its phase sets.
- No internal references in produced files: no IDs of planning decisions or review findings, no section numbers of
  this spec, no `md/` paths, no local paths (e.g. of the TR references folder) and no QuaDRiGa-NG line numbers in any
  source, header, test, wrapper, conf file or documentation block. Cite TR 38.901 (clause, table, equation, note) or
  other published material, or state the rule itself. Existing comments with such labels in a file that a run
  edits are reworded the same way (the rule stays, the label goes).
- No `std::to_chars` for floating point (MATLAB R2024a ships libstdc++ up to `GLIBCXX_3.4.28`). Every run that
  builds MEX files runs the gate `objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u`
  (nothing above 3.4.28).
- No global mutable state; random numbers only from the explicit seed (`lsf_init_phases`) or the caller's values.
- quadriga-lib must not depend on QuaDRiGa-NG: no paths to it, no data derived from its files in sources or tests
  (formulas and tables typed from this spec, the MATLAB sources and the TR are fine). QuaDRiGa-NG and the TR are
  read-only.
- Agents do not commit, push or merge (the maintainer commits). They do not edit this spec; proposals go into the
  run report.

### 9.2 Resources and commands

- Check `cat /proc/loadavg` before heavy work; at most 12 threads (`CMAKE_BUILD_PARALLEL_LEVEL=12` for `make`
  targets, `cmake --build build_linux --parallel 12`, `OMP_NUM_THREADS=12` or less); never two builds or two test
  suites at the same time; heavy tools (ASan, MATLAB, sweeps) once on the final code of a run.
- Catch2: `make test` (configure with tests, build, run all); iterate with
  `cmake --build build_linux --parallel 12 && build_linux/test_bin "lsf_draw*"`; diagnostics with
  `QD_DIAG_GBSM=1 build_linux/test_bin "path_loss*"`.
- MEX: `make matlab=ON` / `make octave=ON`; single tests from MATLAB/Octave as in SPEC_MATLAB_API; MATLAB R2024a,
  R2026a and Octave can be run from the shell (e.g. `matlab -batch "..."`, `octave --eval "..."`; the run report
  states which binaries were used). The MEX suite runs in Octave and R2024a in Run 1 and Run 4a.
- Python: `make python`, `make python_test`, single files with `python3 -m pytest tests/python_tests/test_<name>.py -x -s`.
- No CUDA involved.


## 10. Integration hand-over (QuaDRiGa-NG, after the merge; out of scope for Run 1 to Run 4b)

Starts after the merge into `main`; QuaDRiGa-NG (NG) becomes writable then. The manager splits it into prompts.

### 10.1 Library update in NG

Copy the new quadriga-lib (as in the earlier ports: `external/`, the MEX `+quadriga_lib`), copy the 23 v19 conf
files to `quadriga_src/config`.

### 10.2 Builder LSF

The builder stores `lsf_phase` (`[n_coeff, 11, 2]`) instead of the 8 + 2 `qd_sos` objects in `sos`, `absTOA_sos`,
`gr_sos` (`init_sos` LSF part replaced by `lsf_init_phases`; `copy`, `split_rx`, `split_multi_freq`, `add_paths`
adapted); `gen_lsf_parameters`, `get_sf_profile`, `get_lsp_map` become thin wrappers of `lsf_draw` + `path_loss` +
`lsp_generate`; `lsp_vals` is replaced by the per-link `lsp_mu` / `lsp_sigma` outputs where it is used (`add_sdc`,
`gen_ssf_parameters`; call with `rnd = zeros` where no draw is wanted), or by their mean over the builder's links
where a per-builder value is needed (no reference geometry reproduces `lsp_vals` for files with elevation terms,
section 8); `get_pl` becomes a wrapper of `path_loss`.
The builder passes the floor index of each UT (stored by the layout, 10.3) as `floor_id` to `lsf_draw`, on the drop
positions and on every track position. It stores an explicit seed (drawn once if the user gives none) or the phase
cubes, so that the SSF sets can be drawn independently of the LSF set (section 8). The random hE (`PL_hE_rand`,
UMa) uses column 10 of `lsf_draw`; where steps of the path loss along a track segment are unwanted, the builder
evaluates `rnd_he` once per segment, at its initial position, and uses it for all snapshots of the segment (3.3.6).
The wrapper adds the O2I loss of the track once: today's `nlos` branch of `get_pl.m` adds it twice when the LOS term
wins (once in the recursive `dual_slope` call, once at the end), and `path_loss` never adds O2I loss.

SF standard deviation: today the `sig`, `sig1` to `sig3` and `SF` keys of a path-loss model act only in the
3GPP-baseline branch of `get_channels` (and in `get_los_channels`), which scales the SF by `scale_sf` of `get_pl`;
the default branch takes the SF of `get_sf_profile` (`gen_lsf_parameters`, std from the scenpar keys) unscaled. In
the library the SF std of the path-loss model enters through the `sf_sigma` input of `lsp_generate` (the TR
behaviour; e.g. v19 RMa_LOS and SMa_LOS: 4 or 6 dB by the side of dBP). `get_channels` and `get_los_channels`
therefore drop `scale_sf` and take the SF from `lsp_generate`; whether `sf_sigma` is passed is the switch in the
table below. `simpar` gets the legacy switches with a "legacy" preset:

| Switch | 3GPP default | Legacy (today's QuaDRiGa) |
|---|---|---|
| `lsp_generate` `cholesky` | true | false (symmetric square root, as `sqrtm`) |
| `lsp_generate` `clip_spreads` | true | false |
| `lsf_init_phases` / `lsf_draw` `horizontal` | true | false (3D) |
| `path_loss` `d2d_in` | omitted (geometric distance) | the indoor 2D distance of each link (outdoor distance) |
| `lsp_generate` `sf_sigma` | `sf_sigma` of `path_loss` | omitted (SF std from the scenpar keys, today's default mode; today's baseline mode corresponds to the 3GPP default) |

Changes against today that are not switchable and that NG documents:
- `winner_los` uses `C2` beyond the breakpoint and per-link breakpoint heights (the 5 WINNER LOS files change
  beyond the breakpoint, and where the heights vary within one MATLAB call).
- Bias removal: each Normal field is corrected once over all positions where it is evaluated, and the Tx side is
  mirrored from the corrected phases, so bias-removed ASD/ASA and ESD/ESA change in dual mobility and, for
  different decorrelation distances of a pair, also in single mobility (3.1, "Difference to QuaDRiGa").
- The path-loss constants of the v19 files are recomputed from the TR and rounded to 4 significant digits (changes
  up to 0.014 dB).
- O2I: only the TR models; the deviation is N(0,1) (QuaDRiGa: uniform).
- Dual mobility: ASD/ASA and ESD/ESA are the two-map sums without QuaDRiGa's additional 2x2 square root (3.2
  step 5); the values change in dual mobility.
- Legacy O2I distance per link: `path_loss` computes the indoor 3D distance and the 10 m rule per link from
  `d2d_in`, while `gen_o2i_loss.m` computes `o2i_d3din` once at the start of each track segment and `get_pl.m`
  applies it to every snapshot of the segment; equal at the segment start.
- `sf_sigma` per link: `get_pl.m` averages `sf_sigma` over the snapshots of a segment (`use_track`); the library
  returns it per link, which differs where a segment crosses a breakpoint with `sig1 != sig2`.
- O2I links of `nlos` files: MATLAB returns `max(PLn, PL_LOS + o2i) + o2i`; the library gives
  `max(PLn, PL_LOS)` and the caller adds the O2I loss once.
- `lsp_vals` has no library counterpart (section 8); this changes `add_sdc` and the XPR sigma of
  `gen_ssf_parameters` for files with distance, height or elevation terms.
- A standard deviation that the distance, height or elevation terms make negative is limited at 0 (3.0.4); MATLAB
  flips the sign of the realization there. This affects NG files with `_kappa`, `_tau` or `_beta` terms at the
  links where it happens (e.g. satellite and 5G-ALLSTAR files at low elevation).
- The InF and InH NLOS v19 files bound the absolute TOA by `2L/c` of their calibration hall (4.4.1).

### 10.3 Layout

`set_scenario` keeps a table group -> (LOS model name, member files, parameters) and calls `los_probability`; LOS
and indoor state distances from `LOS_state_lambda`, `IO_state_lambda`; groups for the v19 set incl. SMa and InF-HH;
NG's deprecated 3GPP_3D groups have no model name in the library (NG drops them or maps them to `38.901_UMi` /
`38.901_UMa`). `gen_o2i_loss` is replaced by `o2i_loss` (models and weights from the group,
`d2d_in_max = O2I_d2D_in_max`); the mmMAGIC model, the TR v14 high-loss wall and the legacy d2D-in distribution are
deprecated and stay in MATLAB only if NG still needs them. Order: `o2i_loss` before `los_probability`,
`d2d_out = max(d2D - d2d_in, 0)`; the layout stores `d2d_in` per UT, which the builder passes to `path_loss` in
legacy mode (instead of today's d3D-in). The layout derives the floor index of each UT and stores it: for the TR UT
heights `hUT = 3(n_fl - 1) + 1.5` (TR 38.901 Table 7.2-1, which takes them from TR 36.873) the index is
`round((hUT - 1.5)/3)`, outdoor UTs get 0; the builder passes it to `lsf_draw` (10.2), and the SSF sets use the
same IDs (section 8). An in-car option for UTs (`car`, `car_metal`). The renormalisation of the LOS-state draw stays
in NG as a legacy option.

### 10.4 Parity test, in two parts

A MATLAB script computes, with the pre-integration code, the SOS values `val`, PL, SF and the 8 LSPs of the legacy
files for single mobility at fixed phases and stores them in NG's test data.

1. `lsp_generate` and `path_loss` with the legacy switches, fed with MATLAB's own SOS values as `rnd` (columns 0..7
   from `val`), reproduce PL, SF and the LSPs to 1e-12 relative. The 5 WINNER LOS files (`winner_los`) are compared
   only for links before the breakpoint, since the library uses `C2` beyond it; the breakpoint itself uses per-link
   heights, equal to MATLAB when the heights within one MATLAB call are constant. In legacy O2I mode `path_loss`
   gets `d2d_in` and computes the indoor 3D distance with the geometry of `gen_o2i_loss.m` (3.3.2). The comparison
   uses segment starts or single-position tracks (MATLAB keeps the indoor 3D distance and the averaged `sf_sigma`
   constant per segment, 10.2), and the O2I links of `nlos` files are compared with the O2I loss set to 0 in the
   reference script (MATLAB adds it inside the maximum as well, 10.2). SF: `lsp_generate` without `sf_sigma` is
   compared with the SF of `gen_lsf_parameters` (default mode); with `sf_sigma` from `path_loss` it is compared with
   the baseline value `10·lg(sf)·scale_sf`, for files without `SF_omega`, `SF_kappa`, `SF_tau`, `SF_beta` only
   (`scale_sf` rescales a std that MATLAB computed with these terms). Links where the std limit of 3.0.4 acts are
   left out.
2. `lsf_draw` against MATLAB's `val`: 1e-5 absolute, because `lsf_draw` evaluates with the AVX2 kernel
   (single-precision sin/cos) and the frequencies of `qd_sos` (`freq10 · maxD_old / maxD_new`) are not guaranteed
   to be bitwise equal to those of `sos_preset(..., lambda)`.

The phase cube for the library is built from the old generators: `phase(:, g, 1) = h_builder.sos(g).sos_phase(:,
1)` for g = 1..8 (MATLAB indices), `absTOA_sos` for column 9 (any phases if absent, the value is then unused),
`gr_sos` for column 10, any phases for column 11 (hE; unused, legacy files have no `PL_hE_rand`); slice 2 by
mirroring. The parity runs without bias removal in the library (the phases come from MATLAB, already corrected by
MATLAB's rule), so the bias-removal difference of 3.1 does not enter.

### 10.5 NG unit tests, tutorials, retired code

All `xunit_new` tests pass in MATLAB R2024a and Octave; the calibration tutorials
`calib_3GPP_38901_large_scale.m` and `calib_3GPP_38901_full.m` run with the v19 groups.

Retired MATLAB code (target): `gen_o2i_loss.m` except the deprecated parts NG keeps; the LSF part of `init_sos.m`;
most of `gen_lsf_parameters.m`, `get_pl.m`, `get_sf_profile.m`, `get_lsp_map.m`; the p_LOS formulas in
`set_scenario.m`; the `lsp_vals` property.


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
