// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_quadriga_lib_functions.hpp"

/*!SECTION
Geometry-based stochastic model
SECTION!*/

/*!MD
# CONF_FILE_WRITE
Write a QuaDRiGa scenario configuration file

- Writes scenario parameters, path-loss parameters and short names in the layout of the conf files
  of QuaDRiGa; every written file passes [[conf_file_read]] with `validate = true`, without ignored
  keys and with the same values bit for bit
- The header has the scenario name (the file name without `.conf`), `ref`, the validity range and
  the short names. Up to 10 sections follow, each with an optional line `% See: ref; ref_sec` and,
  for the LSPs and the path-loss model, the formulas as comments; the `plpar` fields are written in
  struct order. Sections by entry of `ref_sec`: 1 large-scale distributions, 2 large-scale
  decorrelation distances, 3 inter-parameter correlations, 4 model parameters, 5 small-scale
  decorrelation distance, 6 ground reflection, 7 absolute TOA, 8 path-loss model, 9 EoD offset, 10
  optional modelling components; 1 to 8 are those of QuaDRiGa, so its 8 section references can be
  passed unchanged
- Numbers are written independent of the locale; `Inf`, `-Inf` and `-0` are kept
- `write_defaults = false`: only fields whose value differs from the default (bit for bit); a
  section without a written field is left out
- All inputs are validated before the file is opened: the checks of [[scenpar_validate]]; `model`,
  the other fields of `plpar` and the short names are names, the short names distinct; every `plpar`
  field has at least one value and no `NaN`; `fn`, `ref` and `ref_sec` have no line break
- A failed call creates no file and leaves an existing file unchanged; the directory must be writable

## Usage:
```
quadriga_lib.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, ...
    write_defaults );
```

## Inputs:
- **`fn`** — Path of the conf file to write; string
- **`scenpar`** — Scenario parameters; struct with any subset of the fields returned by
  [[conf_file_read]]; real scalars of any numeric class or logical; missing fields have their default; `[]` writes
  the default set
- **`plpar`** — Path-loss parameters; struct with the field `model` (string) and one real vector (row or
  column) of any numeric class per further field, named without the prefix `PL_`, as returned by
  [[conf_file_read]]; a matrix is an error; written in struct order; default: no path-loss section
- **`short_names`** — Short names of the scenario; cell array of strings or one string; each one name
  of the characters `A-Z`, `a-z`, `0-9`, `_`; distinct; default: none
- **`ref`** — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; string; default: no reference
- **`ref_sec`** — Reference per section, numbered as listed above; cell array of up to 10 strings;
  an empty or missing entry writes no reference; default: none
- **`valid_range`** — Validity range, written as comment; rows: frequency in GHz, distance, BS
  height, UT height in m; columns: lower and upper end, one value is written if both are equal; a
  row with `NaN` is not written; `[4, 2]`; default: none
- **`write_defaults`** — `true`: write all fields; `false`: write only fields that differ from their
  default; default: false

## See also:
- [[conf_file_read]] (read the file back)
- [[scenpar_validate]] (value checks and defaults)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs;

    // Validate argument counts
    if (nrhs < 2 || nrhs > 8)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read inputs
    const std::string fn = qd_mex_get_string(prhs[0]);

    const auto scenpar = qd_mex_struct2scenpar(prhs[1], "scenpar");

    std::string pl_model;
    std::vector<std::string> pl_order;
    std::unordered_map<std::string, std::vector<double>> plpar;
    if (nrhs > 2)
        plpar = qd_mex_struct2plpar(prhs[2], pl_model, pl_order, "plpar");

    std::vector<std::string> short_names, ref_sec;
    if (nrhs > 3 && !mxIsEmpty(prhs[3]))
    {
        if (!mxIsCell(prhs[3]) && !mxIsChar(prhs[3]))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'short_names' must be a cell array of strings.");
        short_names = qd_mex_get_strings(prhs[3], "short_names");
    }

    const std::string ref = (nrhs < 5) ? std::string() : qd_mex_get_string(prhs[4]);

    if (nrhs > 5 && !mxIsEmpty(prhs[5]))
    {
        if (!mxIsCell(prhs[5]))
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'ref_sec' must be a cell array of strings.");
        ref_sec = qd_mex_get_strings(prhs[5], "ref_sec");
    }

    const arma::mat valid_range = (nrhs < 7) ? arma::mat() : qd_mex_typecast_Mat<double>(prhs[6], "valid_range");
    const bool write_defaults = (nrhs < 8) ? false : qd_mex_get_scalar<bool>(prhs[7], "write_defaults", false);

    // Call library function
    CALL_QD(quadriga_lib::conf_file_write(fn, scenpar, pl_model, &plpar, &pl_order, &short_names,
                                          ref, &ref_sec, &valid_range, write_defaults));
}
