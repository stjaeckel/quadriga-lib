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
- Outside the range of the data sampling points, the nearest end value is returned (constant extrapolation);
  output sampling points that are NaN give NaN.
- The data sampling points need not be sorted; repeated values are allowed; they must be finite.
- Supports single and multiple data sets.
- Data types: `single` or `double`

## Usage:

```
output = quadriga_lib.interp( x, y, input, x_new, y_new );    % 2D case

output = quadriga_lib.interp( x, [], input, x_new );           % 1D case
```

## Inputs:
- **`x`** — Data x-axis sampling points; Length: `[nx]`
- **`y`** — Data y-axis sampling points; Length: `[ny]`
- **`input`** — Input data array/matrix; `[ny, nx, ne]` or `[1, nx, ne]` for 1D case; 3rd dimension
  enables interpolation for multiple datasets simultaneously.
- **`x_new`** — Output x-axis sampling points; Length: `[nx_new]`
- **`y_new`** — Output y-axis sampling points; Length: `[ny_new]`; required when `y` is not empty, empty or omitted
  when `y` is empty

## Output:
- **`output`**  —  Interpolated data `[ny_new, nx_new, ne]` or `[1, nx_new, ne]` for 1D case
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 4 || nrhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    bool twoD = mxGetNumberOfElements(prhs[1]) != 0;
    if (twoD && nrhs < 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'y_new' is required when 'y' is given (2D case).");
    if (!twoD && nrhs > 4 && mxGetNumberOfElements(prhs[4]) != 0)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Input 'y_new' must be empty when 'y' is empty (1D case).");

    if (mxIsSingle(prhs[0]))
    {
        const arma::fvec x = qd_mex_get_Col<float>(prhs[0], false, "x");
        const arma::fvec y = twoD ? qd_mex_get_Col<float>(prhs[1], false, "y") : arma::fvec(1);
        const arma::fcube input = qd_mex_get_Cube<float>(prhs[2], false, "input");
        const arma::fvec x_new = qd_mex_get_Col<float>(prhs[3], false, "x_new");
        const arma::fvec y_new = (nrhs > 4 && twoD) ? qd_mex_get_Col<float>(prhs[4], false, "y_new") : arma::fvec(1);

        arma::fcube output;
        plhs[0] = qd_mex_init_output(&output, y_new.n_elem, x_new.n_elem, input.n_slices);

        CALL_QD(quadriga_lib::interp_2D(x, y, input, x_new, y_new, output));
    }
    else
    {
        const arma::vec x = qd_mex_get_Col<double>(prhs[0], false, "x");
        const arma::vec y = twoD ? qd_mex_get_Col<double>(prhs[1], false, "y") : arma::vec(1);
        const arma::cube input = qd_mex_get_Cube<double>(prhs[2], false, "input");
        const arma::vec x_new = qd_mex_get_Col<double>(prhs[3], false, "x_new");
        const arma::vec y_new = (nrhs > 4 && twoD) ? qd_mex_get_Col<double>(prhs[4], false, "y_new") : arma::vec(1);

        arma::cube output;
        plhs[0] = qd_mex_init_output(&output, y_new.n_elem, x_new.n_elem, input.n_slices);

        CALL_QD(quadriga_lib::interp_2D(x, y, input, x_new, y_new, output));
    }
}
