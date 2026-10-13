// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

#include <algorithm>

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# SCENPAR_VALIDATE
Validate scenario parameters and complete them with default values

- Checks a struct of QuaDRiGa scenario parameters that did not come from a conf file and adds every
  missing field with its default value, so that `scenpar_out` always has all 166 fields
- An empty input (`[]` or a struct without fields) returns the complete default parameter set
- A struct with the fields of QuaDRiGa is completed with the 27 fields for 3GPP TR 38.901 v19.5.0,
  which have no effect at their default of 0
- All fields are real numeric scalars, except `SubpathMethod`, which is a string
- A field that is not one of the 166 names is an error, also with `validate = false`; this includes
  the QuaDRiGa field `GR_epsilon`, which is not supported
- Value checks with `validate = true`; `NaN` is an error for every field:
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
  - All other numeric fields accept any value including `Inf` and `-Inf`
  - `NumClusters_min` must not exceed `NumClusters`
  - `NF_alpha` and `NF_beta` must be both 0 or both > 0
  - `SubpathMethod` must be `legacy`, `Laplacian` or `mmMAGIC`
- With `validate = false` no value is checked: `NaN` and any `SubpathMethod` are accepted
- The symmetric matrix `lsp_xcorr` is built from the 28 correlation fields and has a unit diagonal.
  It is tested by a Cholesky factorization. A matrix that is not positive definite is not an error.

## Usage:
```
[ scenpar_out, lsp_xcorr, lsp_xcorr_chk ] = quadriga_lib.scenpar_validate( scenpar_in, validate );
```

## Inputs:
- **`scenpar_in`** *(optional)* — Scenario parameters; struct with any subset of the 166 fields;
  numeric fields can have any real numeric class; if omitted or empty, the default set is returned
- **`validate`** — `false`: only complete with defaults, no value checks; default: true

## Outputs:
- **`scenpar_out`** — Complete scenario parameters; given values are kept, missing fields have their
  default; struct with 166 fields in the QuaDRiGa field order: the 138 numeric keys of QuaDRiGa with
  `SubpathMethod` (string) as field 107, then the 27 new keys; all numeric fields are double scalars
- **`lsp_xcorr`** — LSP cross-correlation matrix; order of rows and columns: DS, KF, SF, ASD, ASA,
  ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite; `false` also if it contains `NaN`;
  logical

## See also:
- [[conf_file_read]] (read the parameters from a conf file)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs: 1x1 struct of real numeric scalars and the string field 'SubpathMethod'
    std::unordered_map<std::string, double> scenpar;
    std::string subpath_method = "legacy";
    if (nrhs > 0 && !mxIsEmpty(prhs[0]))
    {
        if (!mxIsStruct(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'scenpar_in' must be a 1x1 struct.");

        const int n_fields = mxGetNumberOfFields(prhs[0]);
        for (int f = 0; f < n_fields; ++f)
        {
            const std::string name = mxGetFieldNameByNumber(prhs[0], f);
            const mxArray *data = mxGetFieldByNumber(prhs[0], 0, f);
            if (name == "SubpathMethod")
            {
                if (data == nullptr || !mxIsChar(data))
                    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Field 'SubpathMethod' must be a string.");
                subpath_method = qd_mex_get_string(data);
            }
            else
            {
                if (data == nullptr || mxGetNumberOfElements(data) != 1 || mxIsComplex(data) ||
                    !(mxIsNumeric(data) || mxIsLogical(data)))
                    mexErrMsgIdAndTxt("quadriga_lib:CPPerror", ("Field '" + name + "' must be a real numeric scalar.").c_str());
                scenpar[name] = qd_mex_get_scalar<double>(data, name);
            }
        }
    }
    const bool validate = (nrhs < 2) ? true : qd_mex_get_scalar<bool>(prhs[1], "validate", true);

    // Output allocation
    std::vector<std::string> key_order;
    arma::mat lsp_xcorr;
    bool lsp_xcorr_chk = false;

    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&lsp_xcorr, 8, 8);

    // Wrap optional pointers
    arma::mat *p_lsp_xcorr = lsp_xcorr.empty() ? nullptr : &lsp_xcorr;

    // Call library function
    CALL_QD(lsp_xcorr_chk = quadriga_lib::scenpar_validate(scenpar, subpath_method, p_lsp_xcorr, &key_order, validate));

    // Scenario parameters: numeric keys in canonical order, 'SubpathMethod' at its QuaDRiGa position
    std::vector<std::string> fields = key_order;
    fields.insert(std::find(fields.begin(), fields.end(), "LOS_scatter_radius"), "SubpathMethod");
    plhs[0] = qd_mex_make_struct(fields);
    for (const auto &key : key_order)
        qd_mex_set_field(plhs[0], key, qd_mex_copy2matlab(&scenpar.at(key)));
    qd_mex_set_field(plhs[0], "SubpathMethod", mxCreateString(subpath_method.c_str()));

    if (nlhs > 2)
        plhs[2] = mxCreateLogicalScalar(lsp_xcorr_chk);
}
