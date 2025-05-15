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
# CALC_ROTATION_MATRIX
Calculates a 3x3 rotation matrix from a 3-element orientation vector

## Description:
In linear algebra, a rotation matrix is a transformation matrix that is used to perform a rotation
in Euclidean space. The rotation of a rigid body (or three-dimensional coordinate system with a
fixed origin) is described by a single rotation about some axis. Such a rotation may be uniquely
described by three real-valued parameters. The idea behind Euler rotations is to split the complete
rotation of the coordinate system into three simpler constitutive rotation. This function calculates 
the 3x3 rotation matrix **R** from the intrinsic Euler angles.

## Usage:

```
rotation = quadriga_lib.calc_rotation_matrix( orientation, invert_y_axis, transpose )
```

## Example:

The following example obtains the 3x3 matrix R for a 45 degree rotation around the z-axis:

```
bank    = 0;
tilt    = 0;
heading = 45 * pi/180;
orientation = [ bank; tilt; heading ];
rotation = quadriga_lib.calc_rotation_matrix( orientation );
R = reshape( rotation, 3, 3 );
```

## Input Arguments:
- **`orientation`**<br>
  Euler angles (bank, tilt, head) in [rad]; Single or double precision, Size: `[3, n_row, n_col]`

- **`invert_y_axis`**<br>
  Optional parameter. If set to 1, the rotation around the y-axis is inverted.

- **`transpose`**<br>
  Optional parameter. If set to 1, the output is transposed.

## Output Argument:
- **`rotation`**<br>
  The rotation matrix, i.e. a transformation matrix that is used to perform a rotation in 3D
  Euclidean space; Size: `[9, n_row, n_col]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Input:       orientation      Orientation vectors (rows = bank, tilt, heading) in [rad], Size [3, n_row, n_col]
    //              invert_y_axis    If set to 1, the rotation around the y-axis is inverted
    //              transpose        Transpose the output
    // Output:      rotation         Rotation matrix, Size [9, n_row, n_col]

    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:no_input", "Incorrect number of input arguments.");

    if (nlhs != 1)
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:no_output", "Incorrect number of output arguments.");

    // Read scalar variables
    bool invert_y_axis = nrhs < 2 ? false : qd_mex_get_scalar<bool>(prhs[1], "invert_y_axis");
    bool transpose = nrhs < 3 ? false : qd_mex_get_scalar<bool>(prhs[2], "transpose");

    if (mxIsSingle(prhs[0]))
    {
        arma::fcube rotation;
        const auto orientation = qd_mex_reinterpret_Cube<float>(prhs[0]);
        CALL_QD(rotation = quadriga_lib::calc_rotation_matrix(orientation, invert_y_axis, transpose));
        plhs[0] = qd_mex_copy2matlab(&rotation);
    }
    else if (mxIsDouble(prhs[0]))
    {
        arma::cube rotation;
        const auto orientation = qd_mex_reinterpret_Cube<double>(prhs[0]);
        CALL_QD(rotation = quadriga_lib::calc_rotation_matrix(orientation, invert_y_axis, transpose));
        plhs[0] = qd_mex_copy2matlab(&rotation);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:wrong_type", "Input must be provided in 'single' or 'double' precision.");
}
