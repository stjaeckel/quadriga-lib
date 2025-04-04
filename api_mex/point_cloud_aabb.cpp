// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
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
Site-Specific Simulation Tools
SECTION!*/

/*!MD
POINT_CLOUD_AABB
Calculate the axis-aligned bounding box (AABB) of set of points in 3D-sapce

## Description:
The axis-aligned minimum bounding box (or AABB) for a given set points is its minimum bounding 
box subject to the constraint that the edges of the box are parallel to the (Cartesian)
coordinate axes. Axis-aligned bounding boxes are used as an approximate location of the set of
points.

## Usage:

```
aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_index, vec_size );
```

## Input Arguments:

- **`points`**<br>
  Points in 3D-Cartesian space; Size: [ n_points, 3 ]

- **`sub_cloud_index`** (optional)<br>
  Start indices of the sub-clouds in 0-based notation. If this parameter is not given, the AABB of
  the entire point cloud is returned. Type: uint32; Vector of length `[ n_sub_cloud ]`

- **`vec_size`** (optional)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1. For values > 1,
  the number of rows in the output is increased to a multiple of `vec_size`, padded with zeros.

## Output Argument:

- **`aabb`**<br>
  Axis-aligned bounding box of each sub-cloud. Each box is described by 6 values:
  `[ x_min, x_max, y_min, y_max, z_min, z_max ]`; Size: `[ n_sub_cloud, 6 ]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - points           Points in 3D-Cartesian space; Size: [ n_points, 3 ]
    //  1 - sub_cloud_index  Sub-cloud index, 0-based, Length: [ n_sub ]
    //  2 - vec_size        Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA), default = 1

    // Output:
    //  0 - aabb            Axis-aligned bounding box (AABB) of the point cloud

    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:point_cloud_aabb:IO_error", "Wrong number of input arguments.");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:point_cloud_aabb:IO_error", "Too many output arguments.");

    // Validate data types
    bool use_single = false;
    if (mxIsSingle(prhs[0]) || mxIsDouble(prhs[0]))
        use_single = mxIsSingle(prhs[0]);
    else
        mexErrMsgIdAndTxt("quadriga_lib:point_cloud_aabb:IO_error",
                          "Input 'points' must be provided in 'single' or 'double' precision.");

    arma::fmat points_in_single;
    arma::mat points_in_double;
    if (use_single)
        points_in_single = qd_mex_reinterpret_Mat<float>(prhs[0]);
    else
        points_in_double = qd_mex_reinterpret_Mat<double>(prhs[0]);

    arma::u32_vec sub_cloud_index;
    if (nrhs > 1 && !mxIsEmpty(prhs[1]))
    {
        if (!mxIsUint32(prhs[1]))
            mexErrMsgIdAndTxt("quadriga_lib:point_cloud_aabb:IO_error", "Input 'sub_cloud_index' must be provided as 'uint32'.");

        sub_cloud_index = qd_mex_reinterpret_Col<unsigned>(prhs[1]);
    }

    size_t vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<size_t>(prhs[2], "vec_size", 1);

    // Call the quadriga-lib function
    try
    {
        if (use_single)
        {
            arma::fmat aabb = quadriga_lib::point_cloud_aabb(&points_in_single, &sub_cloud_index, vec_size);
            plhs[0] = qd_mex_copy2matlab(&aabb);
        }
        else
        {
            arma::mat aabb = quadriga_lib::point_cloud_aabb(&points_in_double, &sub_cloud_index, vec_size);
            plhs[0] = qd_mex_copy2matlab(&aabb);
        }
    }
    catch (const std::invalid_argument &ex)
    {
        mexErrMsgIdAndTxt("quadriga_lib:point_cloud_aabb:unknown_error", ex.what());
    }
    catch (...)
    {
        mexErrMsgIdAndTxt("quadriga_lib:point_cloud_aabb:unknown_error", "Unknown failure occurred. Possible memory corruption!");
    }
}