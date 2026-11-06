// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!SECTION_DESC
A **geometry-based stochastic model** (GBSM) generates channels from statistics of a type of environment, the *scenario*
(e.g. urban macro-cell with or without line of sight), instead of from the geometry of one site. The functions of this
section implement 3GPP TR 38.901 clause 7.5 steps 1 to 7 (path loss, LOS probability, O2I penetration loss, large-scale
parameters (LSPs) and clusters) with the spatial consistency of clause 7.6.3. The ray level (steps 8 to 12) follows in a
later version. They are in the submodule `quadriga_lib.gbsm`.<br><br>

**Scenario parameters.** A scenario is a dict of numeric parameters (`scenpar`), a dict of path-loss parameters
(`plpar`) and optional short names, read from a conf file by [[conf_file_read]]. 23 files for TR 38.901 v19.5.0 come
with the library, in `data/config` of the source tree and in `share/quadriga-lib/config` of an installed package (for
pip, below `sys.prefix`); [[conf_file_list]] finds them and [[conf_file_write]] writes them. Every function that takes
`scenpar` accepts the dict of [[conf_file_read]] or [[scenpar_validate]] or any subset of its keys (`None` = all
defaults; a key `SubpathMethod` of QuaDRiGa's structs must be `"legacy"`), completes it with the defaults of missing
keys and validates it with [[scenpar_validate]]; an invalid value of a key that the function does not use is an error
as well. Invalid input raises
`ValueError`, a failure to read or write a file `RuntimeError`.<br><br>

**Workflow.** Per drop: [[o2i_loss]] gives the penetration loss and the indoor distance of each indoor UT, then
[[los_probability]] the LOS probability at the outdoor distance `d2D - d2D-in`. The caller draws the LOS, indoor and O2I
states with [[sos.rand]] / [[sos.randn]] at the correlation distances of TR 38.901 clause 7.6.3.3; the keys
`LOS_state_lambda` and `IO_state_lambda` apply to all links of a scenario and are stated in the NLOS and in the O2I
files (0 in the other files means "not stated here"). Per BS site and scenario: [[gbsm.init_phases]] draws the phases of
each random set once (`"lsf"`; `"cluster"` and `"ray"` with `set_id = -1` and one set per carrier frequency), and
[[gbsm.draw]] evaluates them at the links. Then [[path_loss]] gives the path loss and the shadow-fading (SF) standard
deviation, [[lsp_generate]] the LSPs of each link and frequency, and [[ssf_clusters]] turns the LSPs and the cluster
random values into cluster delays, powers and angles.<br><br>

**Random sets.** The spatially correlated random values come in sets of generators, one column of `rnd` each;
`N = NumClusters - 1`, column `b·N + n` of a `"cluster"` set belongs to block `b` and NLOS cluster `n`:

| `set_type`, `set_id` | `n_gen` | Columns and distributions | Decorrelation distance | Mirror pairs | Used by |
| --- | --- | --- | --- | --- | --- |
| `"lsf"` | 11 | 0 to 7 DS, KF, SF, ASD, ASA, ESD, ESA, XPR, Normal(0,1); 8 absolute TOA, Normal(0,1); 9 ground permittivity, 10 UMa environment height hE, Uniform(0,1) | `DS_lambda` ... `XPR_lambda` (0 to 7), `absTOA_lambda` (8), `SC_lambda` (9), `SF_lambda` (10) | 3 and 4, 5 and 6 | [[lsp_generate]] `rnd`, [[path_loss]] `rnd_he` (column 10) |
| `"cluster"`, -1 | `9·N` | block 0 cluster delay, Uniform(0,1); 1, 2 sign of AOD, AOA; 3, 4 sign of ZOD, ZOA; 5, 6 offset of AOD, AOA; 7, 8 offset of ZOD, ZOA, Normal(0,1) | `SC_lambda` | blocks 1 and 2, 3 and 4, 5 and 6, 7 and 8 | [[ssf_clusters]] `rnd` |
| `"cluster"`, `k` | `N` | cluster shadowing of frequency `k`, Normal(0,1) | `SC_lambda` | none | [[ssf_clusters]] `rnd_freq`, slice `k` |
| `"ray"`, -1 | `14·N·M + N` | blocks 0 to 14 (below) | `SC_lambda`; block 14: $\lambda_\tau$ | below | `ssf_rays` `rnd_ray`, method `"quadriga"` |
| `"ray"`, `k` | `14·N·M + N` | blocks 0 to 14 (below), block 14 unused | `SC_lambda` | below | `ssf_rays` `rnd_ray`, slice `k`, TR methods |

A `"ray"` set holds, with `M = NumSubPaths`, block `b` up to 12 for ray `m` of NLOS cluster `n` in column
`b·N·M + n·M + m`, block 13 in column `13·N·M + n` and block 14 in column `13·N·M + N + n·M + m`; all blocks are
Normal(0,1) except block 14, Uniform(0,1):

| Block | Random value | Mirror pair | Used by |
| --- | --- | --- | --- |
| 0, 1 | coupling rank of AOD, AOA (TR 38.901 clause 7.5 step 8) | 0 and 1 | all methods |
| 2, 3 | coupling rank of ZOD, ZOA | 2 and 3 | all methods |
| 4 | XPR per ray | none | both `pol_model` |
| 5 | initial phase θθ (`pol_model = "quadriga"`: its only phase) | none | both `pol_model` |
| 6, 7 | initial phases θφ, φθ | 6 and 7 | `pol_model = "38.901"` |
| 8 | initial phase φφ | none | `pol_model = "38.901"` |
| 9 | polarisation power variability θθ (TR 38.901 clause 7.6.16) | none | `pol_variability` |
| 10, 11 | the same, θφ and φθ | 10 and 11 | `pol_variability` |
| 12 | the same, φφ | none | `pol_variability` |
| 13 | XPR per cluster (`N` columns) | none | `pol_model = "quadriga"` |
| 14 | ray delay | none | method `"quadriga"` |

Block 14 of the shared `"ray"` set has the decorrelation distance
$\lambda_\tau = 1.6 \cdot c \cdot \bar{c}_\mathrm{DS} / c_\mathrm{ASA}$, limited to 0.1 m to 10 km, with
$\bar{c}_\mathrm{DS}$ the mean cluster delay spread (TR 38.901 Table 7.5-6) over the frequencies passed to
[[gbsm.draw]] and $c_\mathrm{ASA}$ the cluster ASA in rad.

The Tx side of a phase array (slice 1) exchanges the columns of each mirror pair, so that in dual mobility (Tx and Rx
both move) the departure quantities at the Tx are the arrival quantities of the Tx as a second device. A decorrelation
distance of 0 is used as 0.1 m, except for $\lambda_\tau$. Only the `"lsf"` columns 0 to 8 take part in the bias
removal. A set never changes size: new random quantities come in new sets (a new `set_type` or `set_id`).<br><br>

**Conventions.** Positions are in m with the columns x, y, z: `rx_pos` is `(n_link, 3)`, `tx_pos` is `(1, 3)` (one Tx
for all links) or `(n_link, 3)`; heights are the z coordinates. In 2D mode (`horizontal = True`, the TR behaviour) the
random values are correlated in the x-y plane, and `floor_id` separates the floors, which are uncorrelated (TR 38.901
Table 7.5-6 note 3). In 3D mode (`horizontal = False`, legacy QuaDRiGa) the height separates the floors, so UTs on
different floors are correlated by the ACF of their height difference; the per-link statistics are the same in both
modes. Where TR 38.901 and QuaDRiGa differ, the TR behaviour is the default and QuaDRiGa's is available through an
argument where it is still wanted; where QuaDRiGa's behaviour was an error, only the corrected rule is implemented.
Units: frequencies in Hz, path loss in dB, DS in s, KF, SF and XPR as linear power ratios, spreads in degrees.
SECTION_DESC!*/

/*!MD
# conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file has one `Key = value` assignment per line; `%` starts a comment; blank lines, LF or CRLF line ends and a
  UTF-8 byte order mark at the start are accepted; keys are case-sensitive names of `A-Z`, `a-z`, `0-9` and `_`
- Three kinds of keys:
  - 171 numeric scenario parameters (see [[scenpar_validate]]), one number each; keys that are not in the file have
    their default value
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the QuaDRiGa key `GR_epsilon`
  (the complex ground permittivity comes from [[lsp_generate]]). A key stated more than once has its last value
- The line `SubpathMethod = legacy` of QuaDRiGa's files is accepted and ignored; another value of `SubpathMethod` raises
  `ValueError`, also with `validate = False`
- Numbers are decimal with an optional sign, decimal point and exponent (`-6.28`, `.5`, `6.2e-07`), read independent of
  the locale; `Inf`, `+Inf` and `-Inf` in any case; a number below the smallest subnormal reads as 0
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- Raises `RuntimeError` if the file cannot be opened or is a directory
- Raises `ValueError` with file name and line number for a line that is not blank, not a comment and not a well-formed
  assignment (no `=`, not one name before the `=`, a missing value, a word where a number is required or the reverse,
  text after the value, `NaN`, a number beyond the range of a double, a `PL_` key without a name, a `ShortName` value
  stated twice), and with `validate = True` for a value that fails [[scenpar_validate]]

## Usage:
```
scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys = \
    quadriga_lib.gbsm.conf_file_read( fn, validate )
```

## Inputs:
- **`fn`** — Path to the conf file; str, bytes or path-like object (e.g. `pathlib.Path`)
- **`validate`** — `True`: apply the value checks of [[scenpar_validate]]; `False`: only the line
  grammar is checked; bool or `numpy.bool_`; default: True; `None` selects the default

## Outputs:
- **`scenpar`** — Scenario parameters; dict with 171 float entries in the canonical order: the 138
  numeric keys of QuaDRiGa, then the 33 new keys
- **`plpar`** — Path-loss parameters; dict with the entry `model` (str, only if the file states
  `PL_model`) first, then one entry per `PL_` key in file order, named without the prefix `PL_`; each
  is a 1D array with one or more values; `{}` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be
  generated from it; bool
- **`short_names`** — Values of all `ShortName` lines in file order; list of str
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; list of str

## See also:
- [[scenpar_validate]] (value checks and defaults)
- [[conf_file_list]] (find the conf files in a set of directories)
- [[conf_file_write]] (write a conf file)
MD!*/

py::tuple conf_file_read(py::object fn,
                         py::handle validate)
{
    // Read inputs
    const std::string fn_s = qd_python_path(fn, "fn");
    const bool validate_b = qd_python_flag(validate, true, "validate");

    // Output allocation
    std::unordered_map<std::string, double> scenpar;
    std::unordered_map<std::string, std::vector<double>> plpar;
    std::string pl_model;
    std::vector<std::string> key_order, pl_order, short_names, ignored_keys;
    arma::mat lsp_xcorr;

    auto lsp_xcorr_py = qd_python_init_output((arma::uword)8, (arma::uword)8, &lsp_xcorr);

    // Call library function
    const bool lsp_xcorr_chk = quadriga_lib::conf_file_read(fn_s, scenpar, &pl_model, &plpar,
                                                            &lsp_xcorr, &key_order, &pl_order, &short_names,
                                                            &ignored_keys, validate_b);

    // Scenario parameters: numeric keys in canonical order
    auto scenpar_py = qd_python_scenpar2dict(scenpar, key_order);

    // Path-loss parameters: model first, then the numeric keys in file order as 1D arrays
    auto plpar_py = qd_python_plpar2dict(pl_model, plpar, pl_order);

    auto short_names_py = qd_python_copy2list(short_names);
    auto ignored_keys_py = qd_python_copy2list(ignored_keys);

    return py::make_tuple(scenpar_py, plpar_py, lsp_xcorr_py, lsp_xcorr_chk, short_names_py, ignored_keys_py);
}

// pybind11 declaration:
// m.def("conf_file_read", &conf_file_read,
//       py::arg("fn"),
//       py::arg("validate") = true);
