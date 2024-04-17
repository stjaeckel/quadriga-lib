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
#include "mex_helper_functions.cpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# SUBDIVIDE_TRIANGLES
Subdivide the faces of a triangle mesh into smaller faces

## Description:
This function splits the triangles of a mesh into smaller triangles by subdividing the edges
into `n_div` sub-edges. This creates `n_div^2` sub-faces per face.

## Usage:

```
triangles_out = quadriga_lib.subdivide_triangles( triangles_in, no_div );
[ triangles_out, mtl_prop_out ] = quadriga_lib.subdivide_triangles( triangles_in, no_div, mtl_prop_in );
```

## Input Arguments:

- **`triangles_in`**<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; single or double precision;
  <br>Size: `[ n_triangles_in, 9 ]`

- **`no_div`**<br>
  Number of divisions per edge of the input mesh. The resulting number of faces is equal to
  `n_triangles_out = n_triangles_in * n_div^2`

- **`mtl_prop_in`** (optional)<br>
  Material properties of each mesh element; Size: `[ n_triangles_in, 5 ]`

## Output Arguments:

- **`triangles_out`**<br>
  Vertices of the sub-divided mesh in global Cartesian coordinates; singe or double precision;
  <br>Size: `[ n_triangles_out, 9 ]`

- **`mtl_prop_out`**<br>
  Material properties for the sub-divided triangle mesh elements. The values for the new faces are 
  copied from `mtl_prop_in`; Size: `[ n_triangles_out, 5 ]`

MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - triangles_in    Input, matrix of size [n_triangles_in, 9]
    //  1 - n_div           Number of divisions per edge, results in: n_triangles_out = n_triangles_in * n_div^2
    //  2 - mtl_prop_in     Material properties (input), Size: [ n_triangles_in, 5 ], optional (can be empty)

    // Output:
    //  0 - triangles_out   Output, matrix of size [n_triangles_out, 9]
    //  1 - mtl_prop_out    Material properties (output), Size: [ n_triangles_out, 5 ], Empty if input "mtl_prop_in" is empty

    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:IO_error", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:IO_error", "Too many output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:IO_error", "Inputs must be provided in 'single' or 'double' precision of matching type.");

    if (nrhs > 2)
        if ((use_single && !mxIsSingle(prhs[2])) || (!use_single && !mxIsDouble(prhs[2])))
            mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:IO_error", "All floating-point inputs must have the same type: 'single' or 'double' precision");

    // Read inputs
    arma::fmat triangles_in_single, mtl_prop_in_single;
    arma::mat triangles_in_double, mtl_prop_in_double;

    if (use_single)
        triangles_in_single = qd_mex_reinterpret_Mat<float>(prhs[0]);
    else
        triangles_in_double = qd_mex_reinterpret_Mat<double>(prhs[0]);

    if (nrhs > 2 && !mxIsEmpty(prhs[2]))
    {
        if (use_single)
            mtl_prop_in_single = qd_mex_reinterpret_Mat<float>(prhs[2]);
        else
            mtl_prop_in_double = qd_mex_reinterpret_Mat<double>(prhs[2]);
    }

    // Read number of divisions
    arma::uword n_div = (nrhs < 2) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[1], "n_div", 1);

    // Reserve memory for the output
    arma::fmat triangles_out_single, mtl_prop_out_single;
    arma::mat triangles_out_double, mtl_prop_out_double;

    // Call the quadriga-lib function
    try
    {
        if (use_single)
            quadriga_lib::subdivide_triangles(n_div, &triangles_in_single, &triangles_out_single,
                                              &mtl_prop_in_single, &mtl_prop_out_single);
        else
            quadriga_lib::subdivide_triangles(n_div, &triangles_in_double, &triangles_out_double,
                                              &mtl_prop_in_double, &mtl_prop_out_double);
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }

    // Copy data to MATLAB / Octave
    if (nlhs > 0 && use_single)
        plhs[0] = qd_mex_copy2matlab(&triangles_out_single);
    else if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&triangles_out_double);

    if (nlhs > 1 && use_single)
        plhs[1] = qd_mex_copy2matlab(&mtl_prop_out_single);
    else if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&mtl_prop_out_double);
}