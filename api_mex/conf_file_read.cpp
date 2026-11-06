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
A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of
environment, the *scenario* (e.g. urban macro-cell with or without line of sight), instead of from
the geometry of one site. The functions of this section implement 3GPP TR 38.901 clause 7.5 steps 1
to 11 (path loss, LOS probability, O2I penetration loss, large-scale parameters (LSPs), clusters and
rays, without antennas) with the spatial consistency of clause 7.6.3, oxygen absorption (clause
7.6.1) and polarisation power variability (clause 7.6.16); [[gbsm_single_link_channel]] computes the
channel coefficients of a link with its antennas.<br><br>

**Scenario parameters.** A scenario is a struct of numeric parameters (`scenpar`), a struct of
path-loss parameters (`plpar`) and optional short names, read from a conf file by
[[conf_file_read]]. 23 files for TR 38.901 v19.5.0 come with the library, in `data/config` of the
source tree and in `share/quadriga-lib/config` of an installed package; [[conf_file_list]] finds
them and [[conf_file_write]] writes them. Every function that takes `scenpar` accepts the struct of
[[conf_file_read]] or [[scenpar_validate]] or any subset of its fields (`[]` = all defaults; a field
`SubpathMethod` of QuaDRiGa's structs must be `'legacy'`), completes it with the defaults of missing
fields and validates it with [[scenpar_validate]]; an invalid value of a field that the function
does not use is an error as well.<br><br>

**Workflow.** Per drop: [[o2i_loss]] gives the penetration loss and the indoor distance of each
indoor UT, then [[los_probability]] the LOS probability at the outdoor distance `d2D - d2D-in`. The
caller draws the LOS, indoor and O2I states with [[sos_rand]] / [[sos_randn]] at the correlation
distances of TR 38.901 clause 7.6.3.3; the fields `LOS_state_lambda` and `IO_state_lambda` apply to
all links of a scenario and are stated in the NLOS and in the O2I files (0 in the other files means
"not stated here"). Per BS site and scenario: [[gbsm_init_phases]] draws the phases of each random
set once (`'lsf'`; `'cluster'` with `set_id = 0` and one set per carrier frequency; `'ray'` with
`set_id = 0` for the method `'quadriga'`, one set per carrier frequency for the TR methods), and
[[gbsm_draw]] evaluates them at the links. Then [[path_loss]] gives the path loss and the
shadow-fading (SF) standard deviation, [[lsp_generate]] the LSPs of each link and frequency,
[[ssf_clusters]] the cluster delays, powers and angles, and [[ssf_rays]] the rays of each link as a
ray struct (one call for `'quadriga'`, one per frequency for the TR methods). For the mobility
`drift`, [[ssf_scatterers]] places the scatterers of each link and returns `b_med`, which sets the
segment length, and `b_eff`. Per link and segment, [[gbsm_single_link_channel]] computes the
coefficients with `link_gain_dB` = -PL - O2I + 10·lg SF and `kf_ratio` = K(position) / K(drop) from
the large-scale functions; the sum and the ratio are the caller's. Path data read with
[[qrt_file_read]] become a ray struct with [[gbsm_rays_from_paths]]; other path sources fill a ray
struct directly.<br><br>

**Terms.** TR 38.901 names where they exist; QuaDRiGa names are marked:

| Term | Meaning |
| --- | --- |
| scenario | type of environment with its statistics: numeric fields `scenpar`, path-loss model, short names (one conf file) |
| link | one Tx-Rx pair of a batch (row `l` of `rx_pos`) |
| drop | one realisation of LSPs, clusters and rays at the link positions; the positions the ray data refer to |
| cluster | one of the N NLOS clusters of a link: delay, power, angles (TR 38.901 clause 7.5 steps 5 to 7) |
| slot | column of the [[ssf_clusters]] outputs: 1 LOS, 2 GR (if enabled), then the N clusters |
| sub-cluster | one of the three parts of a split cluster (step 11, Table 7.5-5) |
| ray | one of the M rays of a cluster, or the LOS or GR ray (QuaDRiGa: sub-path) |
| path | a propagation path as `get_channels_*` take it; [[ssf_paths]] makes one per kept ray (QuaDRiGa also says path for a cluster) |
| tap | rays summed in the `'taps'` output: the LOS ray, the GR ray, a sub-cluster (TR methods) or a cluster (`'quadriga'`) |
| snapshot | one Tx and Rx position and orientation at which coefficients are computed (QuaDRiGa) |
| segment | the snapshots of one drop over which the ray data are kept (QuaDRiGa) |
| mobility | how the paths follow moved ends within a segment: `phase` (phase rotation per ray, delays and angles fixed) or `drift` (scatterers fixed, geometry recomputed; QuaDRiGa drifting) |
| scatterer | first-bounce (FBS) and last-bounce (LBS) interaction point of a ray; one point for LOS (direct line) and GR (ground) (QuaDRiGa) |
| coefficient power | power of a ray without link gain and oxygen loss (field `power` of the ray struct): the powers of TR 38.901 eq. (7.5-30) split over the rays, GR as [[ssf_clusters]] sets it; for data from paths the path gain |
| path gain | linear power gain of a path incl. link gain and oxygen loss |
| link gain | -PL - O2I + SF in dB of a link at a position (`link_gain_dB`) |
| random set | the random values of one `set_type` and `set_id`, drawn from one phase array; never changes size |
| generator | one spatially consistent random field; one column of `rnd` |
| block | the generators of one random quantity in a set (N or N·M columns) |

**Codes.** Path type: 0 LOS, 1 GR, 2 NLOS. Tap index: 3·(s - 1) + i, with s the column of the
[[ssf_clusters]] outputs and i the sub-cluster (0, 1, 2) of a split cluster and 0 otherwise, so LOS
0, GR 3 and NLOS cluster `n` (from 1) 3·(G + n) + i; `'taps'` lists the taps in ascending index.
Both are codes, not indices: the wrappers do not shift them.<br><br>

**Model properties.**

- Separately dropped UTs keep correlated taps over 43 to 440 wavelengths, while measured fading
  decorrelates within about 0.4 to 1 wavelength: for UTs a few wavelengths apart, use one drop and
  move the UTs (see [[ssf_paths]])
- `'quadriga'`: the per-tap correlation across bands follows the intra-cluster delays of one random
  field; TR methods: independent per band (TR 38.901 clause 7.6.5)
- `pol_model = 'quadriga'` gives a nearly unitary 2x2 cluster matrix (singular-value ratio 0.10 to
  0.22 dB, TR model 7.6 to 8.3 dB)
- The validity of the mobilities `phase` and `drift` is stated in [[ssf_paths]]



**Random sets.** The spatially correlated random values come in sets of generators, one column of
`rnd` each; `N = NumClusters - 1`, columns `b·N + 1` to `(b + 1)·N` of a `'cluster'` set form block
`b`, one column per NLOS cluster:

| `set_type`, `set_id` | `n_gen` | Columns and distributions | Decorrelation distance | Mirror pairs | Used by |
| --- | --- | --- | --- | --- | --- |
| `'lsf'` | 11 | 1 to 8 DS, KF, SF, ASD, ASA, ESD, ESA, XPR, Normal(0,1); 9 absolute TOA, Normal(0,1); 10 ground permittivity, 11 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (1 to 8), `absTOA_lambda` (9), `SC_lambda` (10), `SF_lambda` (11) | 4 and 5, 6 and 7 | [[lsp_generate]] `rnd`, [[path_loss]] `rnd_he` (column 11) |
| `'cluster'`, 0 | `9·N` | block 0 cluster delay, Uniform(0,1); 1, 2 sign of AOD, AOA; 3, 4 sign of ZOD, ZOA; 5, 6 offset of AOD, AOA; 7, 8 offset of ZOD, ZOA, Normal(0,1) | `SC_lambda` | blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 | [[ssf_clusters]] `rnd` |
| `'cluster'`, `k` | `N` | cluster shadowing of frequency `k`, Normal(0,1) | `SC_lambda` | none | [[ssf_clusters]] `rnd_freq`, slice `k` |
| `'ray'`, 0 | `14·N·M + N` | blocks 0 to 14 (below) | `SC_lambda`; block 14: $\lambda_\tau$ | below | [[ssf_rays]] `rnd_ray`, method `'quadriga'` |
| `'ray'`, `k` | `14·N·M + N` | blocks 0 to 14 (below), block 14 unused | `SC_lambda` | below | [[ssf_rays]] `rnd_ray`, slice `k`, TR methods |

A `'ray'` set holds, with `M = NumSubPaths` and ray `m` of NLOS cluster `n` (both from 1), block `b`
up to 12 in column `b·N·M + (n - 1)·M + m`, block 13 in column `13·N·M + n` and block 14 in column
`13·N·M + N + (n - 1)·M + m`; all blocks are Normal(0,1) except block 14, Uniform(0,1):

| Block | Random value | Mirror pair | Used by |
| --- | --- | --- | --- |
| 0, 1 | coupling rank of AOD, AOA (TR 38.901 clause 7.5 step 8) | 0 and 1 | all methods |
| 2, 3 | coupling rank of ZOD, ZOA | 2 and 3 | all methods |
| 4 | XPR per ray | none | both `pol_model` |
| 5 | initial phase θθ (`pol_model = 'quadriga'`: its only phase) | none | both `pol_model` |
| 6, 7 | initial phases θφ, φθ | 6 and 7 | `pol_model = '38.901'` |
| 8 | initial phase φφ | none | `pol_model = '38.901'` |
| 9 | polarisation power variability θθ (TR 38.901 clause 7.6.16) | none | `pol_variability` |
| 10, 11 | the same, θφ and φθ | 10 and 11 | `pol_variability` |
| 12 | the same, φφ | none | `pol_variability` |
| 13 | XPR per cluster (`N` columns) | none | `pol_model = 'quadriga'` |
| 14 | ray delay | none | method `'quadriga'` |

Block 14 of the shared `'ray'` set has the decorrelation distance
$\lambda_\tau = 1.6 \cdot c \cdot \bar{c}_\mathrm{DS} / c_\mathrm{ASA}$, limited to 0.1 m to
10 km, with $\bar{c}_\mathrm{DS}$ the mean cluster delay spread (TR 38.901 Table 7.5-6) over the
frequencies passed to [[gbsm_draw]] and $c_\mathrm{ASA}$ the cluster ASA in rad.

The Tx side of a phase array (slice 2) exchanges the columns of each mirror pair, so that in dual
mobility (Tx and Rx both move) the departure quantities at the Tx are the arrival quantities of the
Tx as a second device. A decorrelation distance of 0 is used as 0.1 m, except for $\lambda_\tau$.
Only the `'lsf'` columns 1 to 9 take part in the bias removal. A set never changes size: new random
quantities come in new sets (a new `set_type` or `set_id`).<br><br>

**Conventions.** Positions are in m with the columns x, y, z: `rx_pos` is `[n_link, 3]`, `tx_pos` is
`[1, 3]` (one Tx for all links) or `[n_link, 3]`; heights are the z coordinates. In 2D mode
(`horizontal = true`, the TR behaviour) the random values are correlated in the x-y plane, and
`floor_id` separates the floors, which are uncorrelated (TR 38.901 Table 7.5-6 note 3). In 3D mode
(`horizontal = false`, legacy QuaDRiGa) the height separates the floors, so UTs on different floors
are correlated by the ACF of their height difference; the per-link statistics are the same in both
modes. Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default and QuaDRiGa's is
available through an argument where it is still wanted; where QuaDRiGa's behaviour was an error,
only the corrected rule is implemented. Units: frequencies in Hz, path loss in dB, DS in s, KF, SF
and XPR as linear power ratios, spreads in degrees.
SECTION_DESC!*/

/*!MD
# CONF_FILE_READ
Read and validate a QuaDRiGa scenario configuration file

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or
  CRLF line ends and a UTF-8 byte order mark at the start are accepted; keys are case-sensitive
  names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - 171 numeric scenario parameters (see [[scenpar_validate]]), one number each; keys that are not
    in the file have their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the
  QuaDRiGa key `GR_epsilon` (the complex ground permittivity comes from [[lsp_generate]]). A key
  stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of
  `SubpathMethod` is an error, also with `validate = false`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`),
  read independent of the locale; `Inf`, `+Inf` and `-Inf` in any case; a number below the smallest
  subnormal reads as 0
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
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
- **`scenpar`** — Scenario parameters; struct with 171 scalar fields in the canonical order: the 138
  numeric keys of QuaDRiGa, then the 33 new keys
- **`plpar`** — Path-loss parameters; struct with the field `model` (string, only if the file states
  `PL_model`) followed by one field per `PL_` key in file order, named without the prefix `PL_`;
  each is a row vector with one or more values; `[]` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be
  generated from it; logical
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
