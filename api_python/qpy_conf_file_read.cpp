// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# conf_file_read
Read and validate a QuaDRiGa scenario configuration file

- A conf file is a text file with one `Key = value` assignment per line; `%` starts a comment; blank
  lines are allowed; line ends can be LF or CRLF; space and tab are white space; a UTF-8 byte order mark at the
  start of the file is skipped
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
  (`e` or `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted in any case
- A number beyond the range of a double (`1e400`) is an error "out of range"; a number below the smallest
  subnormal reads as `0` (or `-0`)
- Numbers are read independent of the locale (the decimal point is always `.`)
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- The 171 numeric keys are 138 keys of QuaDRiGa and 33 keys for 3GPP TR 38.901 v19.5.0. With their
  default of 0 the new keys have no effect.
- The QuaDRiGa key `GR_epsilon` (fixed relative permittivity of the ground) is not supported and is
  reported in `ignored_keys`; the permittivity is complex and is set in the simulation
- Raises `RuntimeError` if the file cannot be opened or is a directory
- Raises `ValueError` with file name and line number for a line that is not blank, not a comment and
  not a well-formed assignment: no `=`, more or less than one name before the `=`, a missing value, a
  word where a number is required or the reverse, a second value or other text after the value, `NaN`,
  complex numbers, a number out of range, a `PL_` key without a name, a `ShortName` value stated twice
- With `validate = True`, a value that fails the checks of [[scenpar_validate]] is an error as well
- A correlation matrix that is not positive definite is not an error; see `lsp_xcorr_chk`

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
- [[scenpar_validate]] (value checks, defaults and the list of validation rules)
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
