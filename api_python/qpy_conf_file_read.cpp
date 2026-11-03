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
`scenpar` accepts the dict of [[conf_file_read]] or [[scenpar_validate]] or any subset of its keys (`SubpathMethod` is
ignored, `None` = all defaults), completes it with the defaults of missing keys and validates it with
[[scenpar_validate]]; an invalid value of a key that the function does not use is an error as well. Invalid input raises
`ValueError`, a failure to read or write a file `RuntimeError`.<br><br>

**Workflow.** Per drop: [[o2i_loss]] gives the penetration loss and the indoor distance of each indoor UT, then
[[los_probability]] the LOS probability at the outdoor distance `d2D - d2D-in`. The caller draws the LOS, indoor and O2I
states with [[sos.rand]] / [[sos.randn]] at the correlation distances of TR 38.901 clause 7.6.3.3; the keys
`LOS_state_lambda` and `IO_state_lambda` apply to all links of a scenario and are stated in the NLOS and in the O2I
files (0 in the other files means "not stated here"). Per BS site and scenario: [[lsf_init_phases]] draws the phases of
the large-scale generators once, [[lsf_draw]] evaluates them at the links, [[path_loss]] gives the path loss and the
shadow-fading (SF) standard deviation, and [[lsp_generate]] the LSPs of each link and frequency. Small scale:
[[ssf_init_phases]] and [[ssf_draw]] do the same for the cluster random values, and [[ssf_clusters]] turns the LSPs and
random values into cluster delays, powers and angles.<br><br>

**Large-scale generators.** Columns of the phase array of [[lsf_init_phases]] and of the random values of [[lsf_draw]];
a decorrelation distance of 0 is used as 0.1 m:

| Column | Generator | Distribution | Decorrelation distance |
| --- | --- | --- | --- |
| 0 to 7 | DS, KF, SF, ASD, ASA, ESD, ESA, XPR | Normal(0,1) | `DS_lambda`, `KF_lambda`, `SF_lambda`, `AS_D_lambda`, `AS_A_lambda`, `ES_D_lambda`, `ES_A_lambda`, `XPR_lambda` |
| 8 | absolute TOA | Normal(0,1) | `absTOA_lambda` |
| 9 | ground permittivity | Uniform(0,1) | `SC_lambda` |
| 10 | UMa environment height hE | Uniform(0,1) | `SF_lambda` |

The Tx side of a phase array (slice 1) exchanges ASD and ASA, ESD and ESA, so that in dual mobility (Tx and Rx both
move) the departure spreads at the Tx are the arrival spreads of the Tx as a second device.<br><br>

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
- Four kinds of keys:
  - 171 numeric scenario parameters (see [[scenpar_validate]]), one number each; keys that are not in the file have
    their default value
  - `SubpathMethod`: one word, default `legacy`
  - `PL_` keys: `PL_model` has one name, the others one or more numbers separated by white space
  - `ShortName`: an alias of the scenario, one name; can occur several times with different values
- Any other key is reported in `ignored_keys` and its value is not examined; this includes the QuaDRiGa key `GR_epsilon`
  (the complex ground permittivity comes from [[lsp_generate]]). A key stated more than once has its last value
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
- **`scenpar`** — Scenario parameters; dict with 172 entries in the QuaDRiGa order: the 138 numeric
  keys of QuaDRiGa with `SubpathMethod` (str) at position 106 (0-based), between `absTOA_lambda` and
  `LOS_scatter_radius`, then the 33 new keys; all other values are float
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
    std::string subpath_method, pl_model;
    std::vector<std::string> key_order, pl_order, short_names, ignored_keys;
    arma::mat lsp_xcorr;

    auto lsp_xcorr_py = qd_python_init_output((arma::uword)8, (arma::uword)8, &lsp_xcorr);

    // Call library function
    const bool lsp_xcorr_chk = quadriga_lib::conf_file_read(fn_s, scenpar, &subpath_method, &pl_model, &plpar,
                                                            &lsp_xcorr, &key_order, &pl_order, &short_names,
                                                            &ignored_keys, validate_b);

    // Scenario parameters: numeric keys in canonical order, 'SubpathMethod' at its QuaDRiGa position
    auto scenpar_py = qd_python_scenpar2dict(scenpar, subpath_method, key_order);

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
