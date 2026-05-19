// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Math functions
SECTION!*/

/*!MD
# CALC_ROTATION_MATRIX
Calculate rotation matrices from Euler angles

- Computes 3×3 rotation matrices from Euler angles (bank, tilt, heading) in column-major order (9 elements
  per orientation)
- Single-precision input is cast to double; output is always double

## Usage:
```
rotation = quadriga_lib.calc_rotation_matrix( orientation, invert_y_axis, transpose );
```

## Inputs:
- **`orientation`** — Euler angles (bank, tilt, heading); `[3, n_row, n_col]` or `[3, n_mat]` or `[3]`
- **`invert_y_axis`** — Flips the sign of the tilt angle, i.e. applies `-tilt` instead of
  `tilt`; use when the input convention defines positive tilt as downward; logical; default: false
- **`transpose`** — Returns the transpose of the rotation matrix; logical; default: false

## Outputs:
- **`rotation`** — Rotation matrices in column-major order; `[9, n_row, n_col]` or `[9, n_mat]` or `[9]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    const arma::cube orientation = qd_mex_get_Cube<double>(prhs[0]);
    const bool invert_y_axis = (nrhs < 2) ? false : qd_mex_get_scalar<bool>(prhs[1], "invert_y_axis", false);
    const bool transpose = (nrhs < 3) ? false : qd_mex_get_scalar<bool>(prhs[2], "transpose", false);

    arma::cube rotation;
    CALL_QD(rotation = quadriga_lib::calc_rotation_matrix<double>(orientation, invert_y_axis, transpose));

    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&rotation);
}