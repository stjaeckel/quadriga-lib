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
 Miscellaneous / Tools
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
```

## Input Arguments:

- **`triangles_in`**<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; singe or double precision;
  <br>Size: `[ n_triangles_in, 9 ]`

- **`no_div`**<br>
  Number of divisions per edge of the input mesh. The resulting number of faces is equal to
  `n_triangles_out = n_triangles_in * n_div^2`

## Output Argument:

- **`triangles_out`**<br>
  Vertices of the sub-divided mesh in global Cartesian coordinates; singe or double precision;
  <br>Size: `[ n_triangles_out, 9 ]`
MD!*/


/*!MATLAB
%SUBDIVIDE_TRIANGLES Subdivides the faces of a triangle mesh into smaller faces
%
% Description:
%   This function splits the triangles of a mesh into smaller triangles by subdividing the edges
%   into 'n_div' sub-edges. This creates 'n_div^2' sub-faces per face.
%
% Input:
%   triangles_in
%   Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
%   points in 3D-space: [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; singe or double precision;
%   Dimensions: [ n_triangles_in, 9 ]
%
%   n_div
%   Number of divisions per edge of the input mesh. The resulting number of faces is equal to
%   n_triangles_out = n_triangles_in * n_div^2
%
% Output:
%   triangles_out
%   Vertices of the sub-divided mesh in global Cartesian coordinates; singe or double precision;
%   Dimensions: [ n_triangles_out, 9 ]
%
%
% quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
% Copyright (C) 2022-2023 Stephan Jaeckel (https://sjc-wireless.com)
% All rights reserved.
%
% e-mail: info@sjc-wireless.com
%
% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
% in compliance with the License. You may obtain a copy of the License at 
% http://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software distributed under the License
% is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
% or implied. See the License for the specific language governing permissions and limitations under
% the License.
MATLAB!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - triangles_in    Input, matrix of size [n_triangles_in, 9]
    //  1 - n_div           Number of divisions per edge, results in: n_triangles_out = n_triangles_in * n_div^2

    // Output:
    //  0 - triangles_out   Output, matrix of size [n_triangles_out, 9]

    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:io_error", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:io_error", "Too many output arguments.");

    // Read inputs
    arma::fmat triangles_in_single;
    arma::mat triangles_in_double;
    if (mxIsSingle(prhs[0]))
        triangles_in_single = qd_mex_typecast_Mat<float>(prhs[0], "triangles_in");
    else if (mxIsDouble(prhs[0]))
        triangles_in_double = qd_mex_typecast_Mat<double>(prhs[0], "triangles_in");
    else
        mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:io_error", "Input must be provided in single or double precision.");

    // Read number of divisions
    unsigned long long n_div = nrhs < 2 ? 1ULL : qd_mex_get_scalar<unsigned long long>(prhs[1], "n_div", 1ULL);

    // Determine number of rows
    unsigned long long n_triangles_in = (mxIsSingle(prhs[0])) ? triangles_in_single.n_rows : triangles_in_double.n_rows;
    unsigned long long n_triangles_out = n_triangles_in * n_div * n_div;

    // Reserver output memory
    if (nlhs > 0)
    {
        try
        {
            if (mxIsSingle(prhs[0]))
            {
                arma::fmat triangles_out;
                plhs[0] = qd_mex_init_output(&triangles_out, n_triangles_out, 9ULL);
                quadriga_lib::subdivide_triangles(n_div, &triangles_in_single, &triangles_out);
            }
            else
            {
                arma::mat triangles_out;
                plhs[0] = qd_mex_init_output(&triangles_out, n_triangles_out, 9ULL);
                quadriga_lib::subdivide_triangles(n_div, &triangles_in_double, &triangles_out);
            }
        }
        catch (const std::invalid_argument &ex)
        {
            mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:unknown_error", ex.what());
        }
        catch (...)
        {
            mexErrMsgIdAndTxt("quadriga_lib:subdivide_triangles:unknown_error", "Unknown failure occurred. Possible memory corruption!");
        }
    }
}