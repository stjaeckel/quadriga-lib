// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# WRITE_PNG
Write a data matrix to a color-coded PNG file

- Values are clipped to `[min_val, max_val]` before colormap mapping; auto-detected from data if `NaN`
- Uses [LodePNG](https://github.com/lvandeve/lodepng) for PNG encoding

## Usage:
```
quadriga_lib.write_png( fn, data, colormap, min_val, max_val, log_transform );
```

## Inputs:
- **`fn`** — Output `.png` file path; string
- **`data`** — Input data matrix; `[n_rows, n_cols]`
- **`colormap`** — Colormap name; supported: jet, parula, winter, hot, turbo, copper, spring, cool, gray, autumn, summer; default: jet
- **`min_val`** — Lower clipping bound; auto-detected if `NaN`; default: `NaN`
- **`max_val`** — Upper clipping bound; auto-detected if `NaN`; default: `NaN`
- **`log_transform`** — Apply 10·log10(data) before mapping; non-positive values map to the minimum color; logical; default: false
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 2 || nrhs > 6)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const std::string fn = qd_mex_get_string(prhs[0]);
    const arma::mat data = qd_mex_get_Mat<double>(prhs[1]);
    const std::string colormap = (nrhs < 3) ? std::string("jet") : qd_mex_get_string(prhs[2], "jet");
    const double min_val = (nrhs < 4) ? NAN : qd_mex_get_scalar<double>(prhs[3], "min_val", NAN);
    const double max_val = (nrhs < 5) ? NAN : qd_mex_get_scalar<double>(prhs[4], "max_val", NAN);
    const bool log_transform = (nrhs < 6) ? false : qd_mex_get_scalar<bool>(prhs[5], "log_transform", false);

    // Call library function
    CALL_QD(quadriga_lib::write_png<double>(data, fn, colormap, min_val, max_val, log_transform));

    // Dummy output for backward compatibility
    double out = 1.0;
    if (nlhs == 1)
        plhs[0] = qd_mex_copy2matlab(&out);
}