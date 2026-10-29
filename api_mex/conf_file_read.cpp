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
the geometry of one site. The functions of this section implement the model of 3GPP TR 38.901,
clause 7.5 steps 1 to 7 (cluster level): path loss, shadow fading, LOS probability, O2I penetration
loss, the large-scale parameters (LSPs) of each link, i.e. the spreads of delays and angles, the
K-factor and the cross-polarization ratio, and the delays, powers and angles of the clusters.<br><br>

The statistics of a scenario are read from a conf file by [[conf_file_read]]. The folder
`data/config` holds 23 files for TR 38.901 v19.5.0; [[scenpar_validate]] completes a parameter set
from another source, [[conf_file_list]] finds the files and [[conf_file_write]] writes them.<br><br>

Per drop (layout): [[o2i_loss]] gives the penetration loss and the indoor distance of each indoor
UT; then [[los_probability]] evaluates the LOS probability at the outdoor distance `d2D - d2D-in`.
The caller draws the LOS state and the indoor state with [[sos_rand]] / [[sos_randn]] at the
decorrelation distances of the fields `LOS_state_lambda` and `IO_state_lambda`, and the O2I
variables at the distances of TR 38.901 clause 7.6.3.3 (see [[o2i_loss]]). The two state distances
apply to all links of a scenario and are stated once in the TR 38.901 files: `LOS_state_lambda` in
the NLOS file, `IO_state_lambda` in the O2I files; 0 in the other files means "not stated here".
Per BS site and scenario (builder): [[lsf_init_phases]] draws the random phases once; [[lsf_draw]]
evaluates them at the links (columns 1 DS, 2 KF, 3 SF, 4 ASD, 5 ASA, 6 ESD, 7 ESA, 8 XPR, 9 absolute TOA: Normal;
10 ground permittivity, 11 UMa environment height: Uniform; each at the decorrelation distance of
its field); [[path_loss]] returns the path loss and the SF standard deviation; [[lsp_generate]]
turns the random values into the LSPs of each link and frequency.<br><br>

The small-scale part implements TR 38.901 clause 7.5 steps 5 to 7 (cluster level). Per BS site and
scenario, [[ssf_init_phases]] draws the phases of the shared cluster set and of one set per
frequency, with seeds derived from the LSF seed; [[ssf_draw]] evaluates them at the links (cluster
delays, signs and angle offsets, and the cluster shadowing per frequency, all at the decorrelation
distance `SC_lambda`); [[ssf_clusters]] turns the LSPs and the random values into the delays, powers
and angles of the clusters with one of three methods: `'38.901'` (clause 7.5, one frequency; the
default), `'38.901_MF'` (the multi-frequency method of clause 7.6.5.1) and `'quadriga'` (spatially
consistent, the same delays and angles at all frequencies, swap-reciprocal). The ray level (steps 8
to 12) follows in a later version.<br><br>

Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default; QuaDRiGa's behaviour is available
through an argument where it is still wanted (3D correlation, symmetric square root of the LSP
cross-correlation matrix, no clipping of the spreads, outdoor path-loss distance of O2I links). Where
QuaDRiGa's behaviour was an error, only the corrected rule is implemented.<br><br>

Units: positions in m, frequencies in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power
ratios, angular spreads in degrees.
SECTION_DESC!*/

/*!MD
# CONF_FILE_READ
Read and validate a QuaDRiGa scenario configuration file

- A conf file is a text file with one `Key = value` assignment per line; `%` starts a comment; blank
  lines are allowed; line ends can be LF or CRLF; space and tab are white space; a UTF-8 byte order
  mark at the start of the file is skipped
- Keys are case-sensitive names of the characters `A-Z`, `a-z`, `0-9` and `_`; there are four kinds:
  - Scenario parameters: 171 numeric keys with one number each. Keys that are not in the file have
    their default value.
  - `SubpathMethod`: one of the words `legacy`, `Laplacian`, `mmMAGIC`; default `legacy`
  - Path-loss parameters: every key that starts with `PL_`. `PL_model` has one name as value. All
    others have one or more numbers separated by white space.
  - `ShortName`: an alias for the scenario, one name; can occur several times with different values
- Any other key is ignored and reported in `ignored_keys`; its value is not examined
- A key that is stated more than once has its last value
- Numbers are decimal with an optional sign, an optional decimal point and an optional exponent
  (`e` or `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted
  in any case
- A number beyond the range of a double (`1e400`) is an error "out of range"; a number below the
  smallest subnormal reads as `0` (or `-0`)
- Numbers are read independent of the locale (the decimal point is always `.`)
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- The 171 numeric keys are 138 keys of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0. With their
  default of 0 the new keys have no effect.
- The QuaDRiGa key `GR_epsilon` (fixed relative permittivity of the ground) is not supported and is
  reported in `ignored_keys`; the permittivity is complex and is set in the simulation
- An error is raised if the file cannot be opened or is a directory, and, with file name and line
  number, for a line that is not blank, not a comment and not a well-formed assignment: no `=`, more
  or less than one name before the `=`, a missing value, a word where a number is required or the
  reverse, a second value or other text after the value, `NaN`, complex numbers, a number out of
  range, a `PL_` key without a name, a `ShortName` value stated twice
- With `validate = true`, a value that fails the checks of [[scenpar_validate]] is an error as well
- A correlation matrix that is not positive definite is not an error; see `lsp_xcorr_chk`

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
- **`scenpar`** — Scenario parameters; struct with 172 fields in the QuaDRiGa field order: the 138
  numeric keys of QuaDRiGa with `SubpathMethod` (string) as field 107, then the 33 new keys; all
  numeric fields are scalars
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
- [[scenpar_validate]] (value checks, defaults and the list of validation rules)
- [[conf_file_list]] (find the conf files in a set of directories)
- [[conf_file_write]] (write a conf file)
- [[lsf_init_phases]] (first step of the large-scale model with the scenario parameters)
- [[path_loss]] (evaluates the path-loss model and its parameters)
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
    std::string subpath_method, pl_model;
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
    CALL_QD(lsp_xcorr_chk = quadriga_lib::conf_file_read(fn, scenpar, &subpath_method, &pl_model, &plpar, p_lsp_xcorr,
                                                         &key_order, &pl_order, p_short_names, p_ignored_keys, validate));

    // Scenario parameters: numeric keys in canonical order, 'SubpathMethod' at its QuaDRiGa position
    plhs[0] = qd_mex_scenpar2struct(scenpar, subpath_method, key_order);

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
