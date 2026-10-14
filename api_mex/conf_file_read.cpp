// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Channel generation functions
SECTION!*/

/*!MD
# CONF_FILE_READ
Read and validate a QuaDRiGa scenario configuration file

- A conf file is a text file with one `Key = value` assignment per line; `%` starts a comment; blank
  lines are allowed; line ends can be LF or CRLF; space and tab are white space
- Keys are case-sensitive names of the characters `A-Z`, `a-z`, `0-9` and `_`; there are four kinds:
  - Scenario parameters: 165 numeric keys with one number each. Keys that are not in the file have
    their default value.
  - `SubpathMethod`: one of the words `legacy`, `Laplacian`, `mmMAGIC`; default `legacy`
  - Path-loss parameters: every key that starts with `PL_`. `PL_model` has one name as value. All
    others have one or more numbers separated by white space.
  - `ShortName`: an alias for the scenario, one name; can occur several times
- Any other key is ignored and reported in `ignored_keys`; its value is not examined
- A key that is stated more than once has its last value
- Numbers are decimal with an optional sign, an optional decimal point and an optional exponent
  (`e` or `E`), for example `-6.28`, `.5`, `6.2e-07`; `Inf`, `+Inf` and `-Inf` are accepted in any case
- Elevation angles are used throughout (+90 deg is up); no key refers to a zenith angle
- The 165 numeric keys are 138 keys of QuaDRiGa and 27 keys for 3GPP TR 38.901 v19.5.0. With their
  default of 0 the new keys have no effect.
- The QuaDRiGa key `GR_epsilon` (fixed relative permittivity of the ground) is not supported and is
  reported in `ignored_keys`; the permittivity is complex and is set in the simulation
- An error is raised if the file cannot be opened, and, with file name and line number, for a line
  that is not blank, not a comment and not a well-formed assignment: no `=`, more or less than one
  name before the `=`, a missing value, a word where a number is required or the reverse, a second
  value or other text after the value, `NaN`, complex numbers, a `PL_` key without a name
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
- **`scenpar`** — Scenario parameters; struct with 166 fields in the QuaDRiGa field order: the 138
  numeric keys of QuaDRiGa with `SubpathMethod` (string) as field 107, then the 27 new keys; all
  numeric fields are scalars
- **`plpar`** — Path-loss parameters; struct with the field `model` (string, only if the file states
  `PL_model`) followed by one field per `PL_` key in file order, named without the prefix `PL_`; each
  is a row vector with one or more values; `[]` if the file has no `PL_` key
- **`lsp_xcorr`** — LSP cross-correlation matrix; symmetric with unit diagonal; order of rows and
  columns: DS, KF, SF, ASD, ASA, ESD, ESA, XPR; `[8, 8]`
- **`lsp_xcorr_chk`** — `true` if `lsp_xcorr` is positive definite, i.e. correlated LSPs can be
  generated from it; logical
- **`short_names`** — Values of all `ShortName` lines in file order; cell array of strings; `[n, 1]`
- **`ignored_keys`** — Unknown keys, each once, in the order of their first occurrence; cell array of
  strings; `[n, 1]`

## See also:
- [[scenpar_validate]] (value checks, defaults and the list of validation rules)
- [[conf_file_list]] (find the conf files in a set of directories)
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
    {
        if (pl_model.empty() && pl_order.empty())
            plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
        else
        {
            std::vector<std::string> fields;
            if (!pl_model.empty())
                fields.push_back("model");
            fields.insert(fields.end(), pl_order.begin(), pl_order.end());
            plhs[1] = qd_mex_make_struct(fields);
            if (!pl_model.empty())
                qd_mex_set_field(plhs[1], "model", mxCreateString(pl_model.c_str()));
            for (const auto &key : pl_order)
            {
                const arma::rowvec values(plpar.at(key));
                qd_mex_set_field(plhs[1], key, qd_mex_copy2matlab(&values));
            }
        }
    }

    if (nlhs > 3)
        plhs[3] = mxCreateLogicalScalar(lsp_xcorr_chk);
    if (nlhs > 4)
        plhs[4] = qd_mex_copy2matlab(&short_names);
    if (nlhs > 5)
        plhs[5] = qd_mex_copy2matlab(&ignored_keys);
}
