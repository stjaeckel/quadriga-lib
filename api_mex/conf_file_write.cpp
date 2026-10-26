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
  shipped with QuaDRiGa, so that [[conf_file_read]] returns the same values bit for bit
- Header: `% Config file for scenario "name"` with the file name without the extension `.conf`,
  then the reference `ref`, the validity range and one line `ShortName = name` per short name
- Up to 10 sections follow, each with a header and, if the entry of `ref_sec` is not empty, a line
  `% See: ref; ref_sec`. Entry in `ref_sec` and write order:
  - 1: Large scale distributions, with an overview formula per LSP; 2: Large-Scale fading
    decorrelation distances; 3: Inter-parameter correlations, with the correlation matrix as
    comment; 4: Model parameters; 9: EoD offset; 5: Decorrelation distance for the small-scale
    fading spatial consistency; 6: Parameters for the ground reflection extension; 7: Absolute
    time of arrival offset; 10: Optional modelling components; 8: Path-loss model
  - The entries 1 to 8 are those of QuaDRiGa, so its 8 section references can be passed unchanged
- The overview formula of an LSP shows its mean and STD with the frequency, distance, height and
  elevation terms; with `LSP_fc_min > 0` the frequency is written as `max( fGHz, fc_min )`; the
  ESD line also has the terms of `ES_D_mu_A` to `ES_D_mu_D` and, if `ES_D_mu_min` is not `-Inf`,
  the lower limit of the mean; the DS line has the term `log10( a · V/S + b )` when `DS_VS > 0`
  (a = `DS_VS_a`, b = `DS_VS_b`; `V/S` stays symbolic)
- Each line is `name = value % comment`, aligned in columns; the comment states the unit
- The path-loss section has the formula blocks of the model (`logdist`, `dual_slope`, `nlos`,
  `tripple_slope`), the line `PL_model`, then the fields of `plpar` in struct order with the prefix
  `PL_`; several values of one field are written on one line
- Numbers are written with up to 15 significant digits, or with 16 or 17 where 15 do not read back
  to the same double in [[conf_file_read]]; `Inf` and `-Inf` are written as such
- With `write_defaults = false` a field is written only if its value differs from the default;
  `SubpathMethod` only if it is not `legacy`; a section without a written field is left out
- With `write_defaults = true` all 171 numeric fields and `SubpathMethod` are written
- The inputs are checked with the rules of [[scenpar_validate]] before the file is opened: unknown
  fields (including `GR_epsilon`), `NaN`, values outside their range and a wrong `SubpathMethod`
  are errors. Further errors: a line break in the file name, a `model` that is not one name, an
  empty or `NaN` path-loss value, short names that are not one name or not distinct, line breaks
  in `ref` or `ref_sec`, more than 10 entries in `ref_sec`, a `valid_range` that is not empty and
  not `[4, 2]`.
- The text is written to a temporary file in the same directory, which must be writable, and then
  renamed to `fn`; a symbolic link `fn` is written through
- A failed call does not create a file and leaves an existing file unchanged; an existing file is
  replaced and keeps its permissions
- Every file written passes [[conf_file_read]] with `validate = true` and without ignored keys

## Usage:
```
quadriga_lib.conf_file_write( fn, scenpar, plpar, short_names, ref, ref_sec, valid_range, ...
    write_defaults );
```

## Inputs:
- **`fn`** — Path of the conf file to write; string
- **`scenpar`** — Scenario parameters; struct with any subset of the 172 fields returned by
  [[conf_file_read]]; numeric fields are real scalars of class double, single, int32, uint32,
  int64, uint64 or logical; `SubpathMethod` is a string; missing fields have their default; `[]`
  writes the default set
- **`plpar`** *(optional)* — Path-loss parameters; struct with the field `model` (string) and one
  real vector (row or column) of class double, single, int32, uint32, int64 or uint64 per further
  field, named without the prefix `PL_`, as returned by [[conf_file_read]]; a matrix is an error;
  written in struct order; if omitted or `[]`, no path-loss section
- **`short_names`** *(optional)* — Short names of the scenario; cell array of strings or one string;
  each one name of the characters `A-Z`, `a-z`, `0-9`, `_`; distinct
- **`ref`** — Literature reference; written in the header and before each non-empty entry of
  `ref_sec`; string; default: no reference
- **`ref_sec`** *(optional)* — Reference per section, numbered as listed above; cell array of up to
  10 strings; an empty or missing entry writes no reference
- **`valid_range`** *(optional)* — Validity range, written as comment; rows: frequency in GHz,
  distance, BS height, MT height in m; columns: lower and upper end, one value is written if both
  are equal; a row with `NaN` is not written; `[4, 2]`
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

    std::string subpath_method = "legacy";
    const auto scenpar = qd_mex_struct2scenpar(prhs[1], subpath_method, "scenpar");

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
    CALL_QD(quadriga_lib::conf_file_write(fn, scenpar, subpath_method, pl_model, &plpar, &pl_order, &short_names,
                                          ref, &ref_sec, &valid_range, write_defaults));
}
