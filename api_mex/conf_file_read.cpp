// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!SECTION_DESC
A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the *scenario*
(e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The functions of this
section implement 3GPP TR 38.901 clause 7.5 steps 2 to 10 and the sub-cluster split of step 11 (LOS probability, path
loss, outdoor-to-indoor (O2I) penetration loss, large-scale parameters (LSPs), clusters and rays, without antennas) with
the spatial consistency of clause 7.6.3, oxygen absorption (clause 7.6.1) and polarization power variability (clause
7.6.16); [[gbsm_single_link_channel]] computes the coefficients of step 11 with the antennas.<br><br>

**Scenario parameters.** A scenario is a struct of numeric fields (`scenpar`), a path-loss struct (`plpar`: the field
`model` and the model parameters) and optional short names, read from a conf file by [[conf_file_read]]. 23 files for
TR 38.901 v19.5.0 come with the library, one per scenario and LOS and indoor state (e.g. `3GPP_38.901_v19_UMa_LOS`,
`_NLOS`, `_LOS_O2I`, `_NLOS_O2I`; InH is `Indoor`, InF NLOS has one file per clutter type), in `data/config` of the
source tree and in `share/quadriga-lib/config` of an installed package; [[conf_file_list]] finds them. Their comments
give the meaning and unit of each key; [[conf_file_write]] with `write_defaults = true` writes the complete default
set with these comments. Every function that takes `scenpar` accepts the struct of [[conf_file_read]] or
[[scenpar_validate]] or any subset of its fields (`[]` = all defaults), completes a copy with the defaults of missing
fields and validates it with [[scenpar_validate]], whose messages pass through, also for fields the function does not
use. A field `SubpathMethod` (QuaDRiGa's structs) must be `'legacy'`.<br><br>

**Workflow** for one drop of a set of links (terms: see Terms below):

1. [[conf_file_read]]: the scenarios, one per LOS and indoor state
2. Per link, by the caller with own SOS generators ([[sos_init_phases]], [[sos_rand]], [[sos_randn]]): the indoor
   state, a Uniform(0,1) value at the decorrelation distance `IO_state_lambda` of the O2I files below an indoor ratio
   of the caller's choice (e.g. TR 38.901 Table 7.2-1), and for indoor UTs [[o2i_loss]] (`loss`, `d2d_in`); then the
   LOS state, a Uniform(0,1) value at `LOS_state_lambda` of the NLOS file below `p_los` of [[los_probability]] at
   `d2d_out = max(d2D - d2d_in, 0)`. The caller chooses the `los_probability` model and the `o2i_loss` models and
   weights (TR 38.901 Tables 7.4.2-1, 7.4.3-2). The states assign each link to a scenario and give the `o2i` flags of
   [[ssf_clusters]]; steps 3 to 6 run per scenario
3. Per BS site and scenario, once, with one seed for all sets: [[gbsm_init_phases]] of the random sets in the table
   below; `'lsf'` with the link positions for the bias removal
4. [[gbsm_draw]] of each set at the links (the shared `'ray'` set with `center_freq`, the list later passed to
   [[ssf_rays]]); [[path_loss]] (`pl`, `sf_sigma`; column 11 of the `'lsf'` set as `rnd_he`); [[lsp_generate]] with
   `sf_sigma`, all seven outputs
5. [[ssf_clusters]] (`n_cluster` is the caller's choice) and [[ssf_rays]] (one ray struct per link), called as in the
   table below
6. Per link and segment: [[ssf_scatterers]] if the mobility is `drift`; then [[gbsm_single_link_channel]] with the
   link gain (see Terms) as `link_gain_dB` (TR methods: once per ray struct, which holds one frequency; `'quadriga'`:
   once per `i_freq`), or [[ssf_paths]] for other channel functions

| Method | Random sets (`set_id`) | Calls in step 5 |
| --- | --- | --- |
| `'38.901'` | `'lsf'`; `'cluster'` 0 and one per frequency; `'ray'` one per frequency | per frequency `k`: [[ssf_clusters]] and [[ssf_rays]] (`i_freq = 1`) with slice `k` of all per-frequency inputs (`center_freq`, `lsp`, `lsp_mu`, `lsp_sigma`, `rnd_freq`, `rnd_ray`, `eod_offset_deg`, `gr_eps_re`, `gr_eps_im`) |
| `'38.901_MF'` | as `'38.901'` | [[lsp_generate]] at the anchor frequency with the same `rnd` gives `lsp_anchor`; [[ssf_clusters]] once; [[ssf_rays]] per frequency `k` with `i_freq = k` |
| `'quadriga'` | `'lsf'`; `'cluster'` 0; `'ray'` 0 | [[ssf_clusters]] and [[ssf_rays]] once for all frequencies |

For moved positions, evaluate [[gbsm_draw]] (`'lsf'`), [[path_loss]], [[lsp_generate]] and for indoor UTs [[o2i_loss]]
again with the same SOS phases: they give the link gain (see Terms) and `kf_ratio` = KF(position) / KF(drop) from
slice 2 of `lsp`. Ray structs can also come from path data, with [[gbsm_rays_from_paths]] (e.g. of [[qrt_file_read]]),
or be filled directly (fields in [[ssf_rays]]); continue with step 6.<br><br>

**Terms.** TR 38.901 names where they exist; QuaDRiGa names are marked:

| Term | Meaning |
| --- | --- |
| scenario | type of environment with its statistics: numeric fields `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx-Rx pair of a batch (row `l` of `rx_pos`) |
| drop | one realization of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 38.901 clause 7.5 steps 5 to 7) |
| slot | column of the [[ssf_clusters]] outputs: 1 LOS, 2 ground reflection (GR, if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (step 11, Table 7.5-5) |
| ray | one of the `NumSubPaths` rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| kept ray | a LOS or GR ray, or an NLOS ray with power > 0 at some frequency (at `i_freq` in [[gbsm_single_link_channel]]) |
| path | a propagation path as `get_channels_*` take it and QRT files store it; [[ssf_paths]] makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in one output tap: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`'quadriga'`) |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting); `auto`: `phase` for the TR methods, `drift` for `'quadriga'` |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (field `power` of the ray struct): the powers of TR 38.901 eq. (7.5-30) split over the rays, GR as [[ssf_clusters]] sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss |
| link gain | `-pl - loss + 10·lg(SF)` in dB of a link at a position and frequency (`link_gain_dB`), with `pl` of [[path_loss]], `loss` of [[o2i_loss]] (0 outdoors) and SF the linear shadow fading, slice 3 of `lsp` |
| random set | the spatially correlated random values of one `set_type` and `set_id` (table in [[gbsm_draw]]) |
| generator | one spatially correlated random field, a [sum-of-sinusoids (SOS)](mex_api.html#sec-sum-of-sinusoids-random-number-generator) generator; one column of a random set |
| SOS phases | the random phases of the sinusoids of the generators of a random set (`phase`); evaluated at positions, they give its random values. "Phase" alone is the phase of a coefficient or ray |
| block | the generators of one random quantity in a set |

**Codes.** Path type: 0 LOS, 1 GR, 2 NLOS. Tap: rays with equal value form one tap; [[ssf_rays]] gives 3·(s - 1) + i,
with s the slot and i the sub-cluster (0, 1, 2) of a split cluster, else 0 (LOS 0, GR 3, the first NLOS cluster 3
without and 6 with GR). Both are labels, not indices: the wrappers do not shift them.<br><br>

**Model properties.** UTs closer than the decorrelation distance `SC_lambda` get correlated clusters and rays, so
separately dropped UTs a few wavelengths apart have unrealistically correlated fading: drop once and move the UTs with
[[ssf_paths]] or [[gbsm_single_link_channel]]. Across bands, `'quadriga'` uses one ray set, so the fading is
correlated; the TR methods draw coupling, XPR and phases per frequency, so it is independent, also with
`'38.901_MF'`, which keeps the delays and angles.<br><br>

**Conventions.** `N = NumClusters - 1` is the number of NLOS clusters: `NumClusters` counts the LOS slot, as in
QuaDRiGa. Positions are in m; in the batch functions one row (x, y, z) per link: `rx_pos` is `[n_link, 3]`, `tx_pos` is
`[1, 3]` (one Tx for all links) or `[n_link, 3]`; heights are the z coordinates. The one-link functions take a position
as a `[3]` vector, except [[gbsm_single_link_channel]], which takes one row per snapshot; scatterer positions are
`[3, n]`. Angles are elevations, not TR zenith angles (EOD, EOA, ESD for ZOD, ZOA, ZSD; also in the conf-file keys): in
rad in the GCS, azimuth in [-pi, pi), elevation in [-pi/2, pi/2] with +pi/2 up. Defaults follow TR 38.901 except
two deliberate library choices: the SOS preset `Comb256` with bias removal ([[gbsm_init_phases]], [[gbsm_draw]]) and
`method = 'quadriga'` of [[gbsm_rays_from_paths]]; QuaDRiGa variants are options. Units: frequencies in Hz, path loss
in dB, DS in s, KF, SF and XPR as linear power ratios, spreads and `_deg` offsets in degrees; conf-file keys and
`scenpar` fields have the units of the file comments (e.g. `PerClusterDS` in ns). Messages name the argument
`center_freq` `center_frequency`.
SECTION_DESC!*/

/*!MD
# CONF_FILE_READ
Read and validate a QuaDRiGa scenario configuration file

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or
  CRLF line ends and a UTF-8 byte order mark at the start are accepted; keys are case-sensitive
  names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - the numeric scenario parameters (see [[scenpar_validate]]), one number each; keys that are not
    in the file have their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the
  QuaDRiGa key `GR_epsilon` (the complex ground permittivity comes from [[lsp_generate]]). A key
  stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of
  `SubpathMethod` is an error, also with `validate = false`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`),
  read independent of the locale; `Inf`, `+Inf` and `-Inf` in any case
- An error is raised if the file cannot be opened or is a directory
- An error with file name and line number for a line that is not blank, not a comment and not a
  well-formed assignment (no `=`, not one name before the `=`, a missing value, a word where a
  number is required or the reverse, text after the value, `NaN`, a number beyond the range of a
  double, a `PL_` key without a name, a `ShortName` value stated twice), and with `validate = true`
  for a value that fails [[scenpar_validate]]

## Usage:
```
[ scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys ] = ...
    quadriga_lib.conf_file_read( fn, validate );
```

## Inputs:
- **`fn`** — Path to the conf file; string
- **`validate`** — `true`: apply the value checks of [[scenpar_validate]]; `false`: only the line
  grammar is checked; default: true

## Outputs:
- **`scenpar`** — Scenario parameters; struct with all scenario fields as double scalars in the
  canonical order of [[scenpar_validate]]
- **`plpar`** — Path-loss parameters; struct with the field `model` (string, only if the file states
  `PL_model`) followed by one field per `PL_` key in file order, named without the prefix `PL_`;
  each is a row vector with one or more values; `[]` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite; `false` is no error here, but [[lsp_generate]]
  raises an error for the scenario; logical
- **`short_names`** — Values of all `ShortName` lines in file order; cell array of strings;
  `[n_short, 1]`
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; cell array
  of strings; `[n_ignored, 1]`

## See also:
- [[scenpar_validate]] (value checks and defaults)
- [[conf_file_list]] (find the conf files in a set of directories)
- [[conf_file_write]] (write a conf file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0]);
    const bool validate = (nrhs < 2) ? true : qd_mex_get_scalar<bool>(prhs[1], "validate", true);

    // Output allocation
    std::unordered_map<std::string, double> scenpar;
    std::unordered_map<std::string, std::vector<double>> plpar;
    std::string pl_model;
    std::vector<std::string> key_order, pl_order, short_names, ignored_keys;
    arma::mat lsp_xcorr;
    bool lsp_xcorr_chk = false;

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&lsp_xcorr, 8, 8);

    // Wrap optional pointers
    arma::mat *p_lsp_xcorr = lsp_xcorr.empty() ? nullptr : &lsp_xcorr;
    std::vector<std::string> *p_short_names = (nlhs > 4) ? &short_names : nullptr;
    std::vector<std::string> *p_ignored_keys = (nlhs > 5) ? &ignored_keys : nullptr;

    // Call library function
    CALL_QD(lsp_xcorr_chk = quadriga_lib::conf_file_read(fn, scenpar, &pl_model, &plpar, p_lsp_xcorr,
                                                         &key_order, &pl_order, p_short_names, p_ignored_keys, validate));

    // Scenario parameters: numeric keys in canonical order
    plhs[0] = qd_mex_scenpar2struct(scenpar, key_order);

    // Path-loss parameters: model first, then the numeric keys in file order as row vectors
    if (nlhs > 1)
        plhs[1] = qd_mex_plpar2struct(pl_model, plpar, pl_order);

    if (nlhs > 3)
        plhs[3] = mxCreateLogicalScalar(lsp_xcorr_chk);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&short_names);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&ignored_keys);
}
