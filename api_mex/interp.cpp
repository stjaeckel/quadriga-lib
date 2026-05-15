// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_math.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# INTERP
Perform linear interpolation (1D or 2D) on single or multiple data sets

- Interpolates given input data at specified output points.
- Supports single and multiple data sets.
- Returns interpolated results either directly or through reference argument.
- Data types: `single` or `double`

## Usage:

```
dataI = quadriga_lib.interp( x, y, data, xI, yI );      % 2D case

dataI = quadriga_lib.interp( x, [], data, xI );         % 1D case
```

## Inputs:
- **`x`** — Data x-axis sampling points; Length: `[nx]`
- **`y`** — Data y-axis sampling points; Length: `[ny]`
- **`data`** — Input data array/matrix; `[ny, nx, ne]` or `[1, nx, ne]` for 1D case; 3rd dimension
  enables interpolation for mutiple datasets simultaneously.
- **`xI`** — Output x-axis sampling points; Length: `[nxI]`
- **`yI`** — Output y-axis sampling points; Length: `[nyI]`

## Output:
- **`dataI`**  —  Interpolated data `[nyI, nxI, ne]` or `[1, nxI, ne]` for 1D case
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    bool twoD = mxGetNumberOfElements(prhs[1]) != 0;

    if (mxIsSingle(prhs[0]) && nlhs > 0)
    {
        const arma::fvec xi = qd_mex_get_Col<float>(prhs[0]);
        const arma::fvec yi = twoD ? qd_mex_get_Col<float>(prhs[1]) : arma::fvec(1);
        const arma::fcube data = qd_mex_get_Cube<float>(prhs[2]);
        const arma::fvec xo = qd_mex_get_Col<float>(prhs[3]);
        const arma::fvec yo = (nrhs > 4 && twoD) ? qd_mex_get_Col<float>(prhs[4]) : arma::fvec(1);

        arma::fcube output;
        plhs[0] = qd_mex_init_output(&output, yo.n_elem, xo.n_elem, data.n_slices);

        CALL_QD(quadriga_lib::interp_2D(data, xi, yi, xo, yo, output));
    }
    else if (nlhs > 0)
    {
        const arma::vec xi = qd_mex_get_Col<double>(prhs[0]);
        const arma::vec yi = twoD ? qd_mex_get_Col<double>(prhs[1]) : arma::vec(1);
        const arma::cube data = qd_mex_get_Cube<double>(prhs[2]);
        const arma::vec xo = qd_mex_get_Col<double>(prhs[3]);
        const arma::vec yo = (nrhs > 4 && twoD) ? qd_mex_get_Col<double>(prhs[4]) : arma::vec(1);

        arma::cube output;
        plhs[0] = qd_mex_init_output(&output, yo.n_elem, xo.n_elem, data.n_slices);

        CALL_QD(quadriga_lib::interp_2D(data, xi, yi, xo, yo, output));
    }
}
