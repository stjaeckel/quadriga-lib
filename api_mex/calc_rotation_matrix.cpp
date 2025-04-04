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
rotation of the coordinate system into three simpler constitutive rotations, called precession, 
nutation, and intrinsic rotation, being each one of them an increment on one of the Euler angles. In 
aviation orientation of the aircraft is usually expressed as intrinsic Tait-Bryan angles following 
the z-y-x convention, which are called heading, tilt, and bank (or synonymously, yaw, pitch, 
and roll). This function calculates the 3x3 rotation matrix **R** from the intrinsic Tait-Bryan 
(orientation) angles. 

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
  This 3-element vector describes the orientation of the array antenna or of individual array elements.
  The The first value describes the ”bank angle”, the second value describes the  ”tilt angle”, 
  (positive values point upwards), the third value describes the bearing or ”heading angle”, in 
  mathematic sense. Values must be given in [rad]. East corresponds to 0, and the angles increase 
  counter-clockwise, so north is pi/2, south is -pi/2, and west is equal to pi. <br>
  Single or double precision, Size: `[3, n_row, n_col]`

- **`invert_y_axis`**<br>
  Optional parameter. If set to 1, the rotation around the y-axis is inverted.

- **`transpose`**<br>
  Optional parameter. If set to 1, the output is transposed.

## Output Argument:

- **`rotation`**<br>
  The rotation matrix, i.e. a transformation matrix that is used to perform a rotation in 3D 
  Euclidean space. The matrix produces the desired effect only if it is used to premultiply column 
  vectors. The rotations are applies in the order: heading (around z axis), tilt (around y axis) 
  and bank (around x axis). The 9 elements of the rotation matrix are returned in column-major 
  order. Single or double precision (same as input), Size: `[9, n_row, n_col]`
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

    if (mxGetNumberOfElements(prhs[0]) == 0)
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:empty", "Input cannot be empty.");

    unsigned n_dim = (unsigned)mxGetNumberOfDimensions(prhs[0]); // Number of dimensions in "orientation"
    const mwSize *dims = mxGetDimensions(prhs[0]);               // Read number of elements elements per dimension
    unsigned n_row = (unsigned)dims[1];                          // Number of rows
    unsigned n_col = n_dim < 3 ? 1 : (unsigned)dims[2];          // Number of columns

    if ((unsigned)dims[0] != 3)
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:size_mismatch", "Input must have 3 elements on the first dimension.");

    // Read scalar variables
    bool invert_y_axis = nrhs < 2 ? false : qd_mex_get_scalar<bool>(prhs[1], "invert_y_axis");
    bool transpose = nrhs < 3 ? false : qd_mex_get_scalar<bool>(prhs[2], "transpose");

    mwSize dims_out[3] = {9, n_row, n_col};
    if (mxIsSingle(prhs[0]))
    {
        const arma::fcube orientation = arma::fcube((float *)mxGetData(prhs[0]), 3, n_row, n_col, false, true);
        arma::cube Rd = quadriga_lib::calc_rotation_matrix(orientation, invert_y_axis, transpose); // double precision output
        arma::fcube R = arma::conv_to<arma::fcube>::from(Rd);                                        // conversion to single
        plhs[0] = mxCreateNumericArray(3, dims_out, mxSINGLE_CLASS, mxREAL);
        std::memcpy((float *)mxGetData(plhs[0]), R.memptr(), sizeof(float) * R.n_elem);
    }
    else if (mxIsDouble(prhs[0]))
    {
        const arma::cube orientation = arma::cube((double *)mxGetData(prhs[0]), 3, n_row, n_col, false, true);
        arma::cube R = quadriga_lib::calc_rotation_matrix(orientation, invert_y_axis, transpose);
        plhs[0] = mxCreateNumericArray(3, dims_out, mxDOUBLE_CLASS, mxREAL);
        std::memcpy((double *)mxGetData(plhs[0]), R.memptr(), sizeof(double) * R.n_elem);
    }
    else
        mexErrMsgIdAndTxt("quadriga_lib:calc_rotation_matrix:wrong_type", "Input must be provided in 'single' or 'double' precision.");
}
