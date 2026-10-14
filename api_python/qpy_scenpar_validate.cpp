// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_quadriga_adapter.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# scenpar_validate
Validate scenario parameters and complete them with default values

- Checks a dict of QuaDRiGa scenario parameters that did not come from a conf file and adds every
  missing key with its default value, so that `scenpar_out` always has all 166 entries
- `None` or an empty dict returns the complete default parameter set
- A dict with the keys of QuaDRiGa is completed with the 27 keys for 3GPP TR 38.901 v19.5.0, which
  have no effect at their default of 0
- All values are real scalars, except `SubpathMethod`, which is a str
- A key that is not one of the 166 names is an error, also with `validate = False`; this includes
  the QuaDRiGa key `GR_epsilon`, which is not supported
- Value checks with `validate = True`; `NaN` is an error for every key:
  - Integer and > 0: `NumClusters`, `NumSubPaths`
  - Finite and > 0: `r_DS`
  - Finite and >= 0: the 8 decorrelation distances (`DS_lambda` ... `XPR_lambda`), `SC_lambda`,
    `absTOA_lambda`, `PerClusterDS`, `PerClusterDS_min`, `PerClusterAS_D`, `PerClusterAS_A`,
    `PerClusterES_D`, `PerClusterES_A`, `LNS_ksi`, `LOS_scatter_radius`, `GR_enabled`, `LSP_fc_min`,
    `EoD_offset_b`, `EoD_offset_hcl`, `PerClusterDS_fc_min`, `PerClusterES_D_scale`, `absTOA_Lmax`,
    `NF_alpha`, `NF_beta`, `SNS_sigma`, `SNS_A`, `SNS_B`, `SNS_R`, `SNS_delta_var`, `PolVar_sigma`
  - Integer and >= 0: `NumClusters_min`, `NF_k1`
  - In the range from 0 to 1: `SNS_mu`
  - In the range from -1 to 1: the 28 correlations (`asD_ds` to `xpr_esa`)
  - All other numeric keys accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `SubpathMethod` must be `legacy`, `Laplacian` or `mmMAGIC`
- With `validate = False` no value is checked: `NaN` and any `SubpathMethod` are accepted
- The symmetric matrix `lsp_xcorr` is built from the 28 correlation keys and has a unit diagonal.
  It is tested by a Cholesky factorization. A matrix that is not positive definite is not an error.
- Raises `ValueError` for invalid input; `scenpar_in` is not modified

## Usage:
```
scenpar_out, lsp_xcorr, lsp_xcorr_chk = quadriga_lib.tools.scenpar_validate( scenpar_in, validate )
```

## Inputs:
- **`scenpar_in`** — Scenario parameters; dict with any subset of the 166 keys returned by
  [[conf_file_read]]; numeric values are real scalars (int, float, bool or a numpy value with one
  element); `SubpathMethod` is a str; `None` or `{}` returns the default set; default: None
- **`validate`** — `False`: only complete with defaults, no value checks; default: True

## Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing keys have their
  default; dict with 166 entries in the QuaDRiGa order: the 138 numeric keys of QuaDRiGa with
  `SubpathMethod` (str) at position 106 (0-based), then the 27 new keys; all other values are float
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `(8, 8)`
- **`lsp_xcorr_chk`** — `True` if `lsp_xcorr` is positive definite; `False` also if it contains
  `NaN`; bool

## See also:
- [[conf_file_read]] (read the parameters from a conf file)
MD!*/

py::tuple scenpar_validate(py::handle scenpar_in,
                           bool validate)
{
    // Read inputs: dict of real scalars and the str entry 'SubpathMethod'
    std::string subpath_method;
    auto scenpar = qd_python_dict2scenpar(scenpar_in, subpath_method, "scenpar_in");

    // Output allocation
    std::vector<std::string> key_order;
    arma::mat lsp_xcorr;

    auto lsp_xcorr_py = qd_python_init_output((arma::uword)8, (arma::uword)8, &lsp_xcorr);

    // Call library function; completes 'scenpar' in place (a copy, the input dict is not changed)
    const bool lsp_xcorr_chk = quadriga_lib::scenpar_validate(scenpar, subpath_method, &lsp_xcorr, &key_order,
                                                              validate);

    // Scenario parameters: numeric keys in canonical order, 'SubpathMethod' at its QuaDRiGa position
    auto scenpar_out_py = qd_python_scenpar2dict(scenpar, subpath_method, key_order);

    return py::make_tuple(scenpar_out_py, lsp_xcorr_py, lsp_xcorr_chk);
}

// pybind11 declaration:
// m.def("scenpar_validate", &scenpar_validate,
//       py::arg("scenpar_in") = py::none(),
//       py::arg("validate") = true);
