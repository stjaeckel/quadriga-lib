// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Miscellaneous / Tools
SECTION!*/

/*!MD
# INTERP
2D and 1D linear interpolation.

## Description:
This function implements 2D and 1D linear interpolation.

## Usage:

```
dataI = quadriga_lib.interp( x, y, data, xI, yI );      % 2D case

dataI = quadriga_lib.interp( x, [], data, xI );         % 1D case
```

## Input Arguments:
- **`x`**<br>
  Vector of sample points in x direction for which data is provided; single or double; Length: `[nx]`

- **`y`**<br>
  Vector of sample points in y direction for which data is provided; single or double; Length: `[ny]`<br>
  Must be an empty array `[]` in case of 1D interpolation.

- **`data`**<br>
  The input data tensor; single or double; Size: `[ny, nx, ne]` or `[1, nx, ne]` for 1D case <br>
  The 3rd dimension enables interpolation for mutiple datasets simultaneously.

- **`xI`**<br>
  Vector of sample points in x direction for which data should be interpolated; single or double;
  Length: `[nxI]`

- **`yI`**<br>
  Vector of sample points in y direction for which data should be interpolated; single or double;
  Length: `[nyI]`

## Output Arguments:
- **`dataI`**<br>
  The interpolated dat; single or double (same as `data`);
  Size: `[nyI, nxI, ne]` or `[1, nxI, ne]` for 1D case <br>
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Need at least 4 input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments.");

    bool twoD = mxGetNumberOfElements(prhs[1]) != 0;

    if (mxIsSingle(prhs[0]) && nlhs > 0)
    {
        const arma::fvec xi = qd_mex_get_single_Col(prhs[0]);
        const arma::fvec yi = twoD ? qd_mex_get_single_Col(prhs[1]) : arma::fvec(1);
        const arma::fcube data = qd_mex_get_single_Cube(prhs[2]);
        const arma::fvec xo = qd_mex_get_single_Col(prhs[3]);
        const arma::fvec yo = (nrhs > 4 && twoD)  ? qd_mex_get_single_Col(prhs[4]) : arma::fvec(1);

        arma::fcube output;
        plhs[0] = qd_mex_init_output(&output, yo.n_elem, xo.n_elem, data.n_slices);

        CALL_QD(quadriga_lib::interp_2D(data, xi, yi, xo, yo, output));
    }
    else if (nlhs > 0)
    {
        const arma::vec xi = qd_mex_get_double_Col(prhs[0]);
        const arma::vec yi = twoD ? qd_mex_get_double_Col(prhs[1]) : arma::vec(1);
        const arma::cube data = qd_mex_get_double_Cube(prhs[2]);
        const arma::vec xo = qd_mex_get_double_Col(prhs[3]);
        const arma::vec yo = (nrhs > 4 && twoD) ? qd_mex_get_double_Col(prhs[4]) : arma::vec(1);

        arma::cube output;
        plhs[0] = qd_mex_init_output(&output, yo.n_elem, xo.n_elem, data.n_slices);

        CALL_QD(quadriga_lib::interp_2D(data, xi, yi, xo, yo, output));
    }
}
