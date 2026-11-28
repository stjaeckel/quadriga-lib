// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Sum-of-sinusoids random number generator
SECTION!*/

/*!MD
# SOS_ACFI
Target ACF at given distances

- Linear interpolation of the autocorrelation function (ACF) given as a table
- Distances beyond the last sample point return the last value of the table

## Usage:
```
val = quadriga_lib.sos_acfi( acf, dist, d );
```

## Inputs:
- **`acf`** — Target ACF; at least 2 elements; `[n_acf]`
- **`dist`** — Sample points of the ACF; must start at 0, increase strictly and be finite; `[n_acf]`
- **`d`** — Distances; must be >= 0; `[n_d]`

## Output:
- **`val`** — ACF at the distances `d`; `[n_d, 1]`

## See also:
- [[sos_preset]] (ACF tables of the predefined models)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const arma::vec acf = qd_mex_get_Col<double>(prhs[0], false, "acf");
    const arma::vec dist = qd_mex_get_Col<double>(prhs[1], false, "dist");
    const arma::vec d = qd_mex_get_Col<double>(prhs[2], false, "d");

    arma::vec val;
    CALL_QD(val = quadriga_lib::sos_acfi<double>(acf, dist, d));
    plhs[0] = qd_mex_copy2matlab(&val);
}
