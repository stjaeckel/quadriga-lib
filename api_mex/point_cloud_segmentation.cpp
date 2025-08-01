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
# POINT_CLOUD_SEGMENTATION
Rearranges elements of a point cloud into smaller sub-clouds

## Description:
This function processes a large 3D point cloud by clustering closely spaced points and recursively 
partitioning it into smaller sub-clouds, each below a specified size threshold. It minimizes the 
axis-aligned bounding box of each sub-cloud while striving to maintain a target number of points 
per cluster.<br><br>

Sub-clouds are aligned to a specified SIMD vector size (e.g., for AVX or CUDA), with padding applied 
as needed. The function outputs a reorganized version of the input points (pointsR), where points 
are grouped by sub-cloud, and provides forward and reverse index maps to track the reordering. This 
organization is particularly useful for optimizing spatial processing tasks such as bounding volume 
hierarchies or GPU batch execution.

## Usage:

```
[ points_out, sub_cloud_index, forward_index, reverse_index ] = ...
    quadriga_lib.point_cloud_segmentation( points_in, target_size, vec_size );
```

## Input Arguments:

- **`points_in`**<br>
  Points in 3D-Cartesian space; Size: `[ n_points_in, 3 ]`

- **`target_size`** (optional)<br>
  The target number of elements of each sub-cloud. Default value = 1024. For best performance, the
  value should be around 10 * sgrt( n_points_in )

- **`vec_size`** (optional)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1.
  For values > 1,the number of rows for each sub-cloud in the output is increased to a multiple
  of `vec_size`. For padding, zero-sized triangles are placed at the center of the AABB of
  the corresponding sub-cloud.

## Output Arguments:

- **`points_out`**<br>
  Points in 3D-Cartesian space; singe or double precision;  Size: `[ n_points_out, 3 ]`

- **`sub_cloud_index`**<br>
  Start indices of the sub-clouds in 0-based notation. Type: uint32; Vector of length `[ n_sub_cloud ]`

- **`forward_index`**<br>
  Indices for mapping elements of "points_in" to "points_out"; 1-based;
  Length: `[ n_points_out ]`; For `vec_size > 1`, the added elements not contained in the input
  are indicated by zeros.

- **`reverse_index`**<br>
  Indices for mapping elements of "points_out" to "points_in"; 1-based; Length: `[ n_points_in ]`

MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    //  0 - points_in       Points in 3D-Cartesian space (input), Size: [ n_points_in, 3 ]
    //  1 - target_size     Target value for the sub-cloud size, default = 1024
    //  2 - vec_size        Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA), default = 1

    // Output:
    //  0 - points_out      Reorganized point cloud (output), Size: [ n_points_out, 9 ]
    //  1 - sub_cloud_index Sub-cloud index, 0-based, Length: [ n_sub ]
    //  2 - forward_index   Indices for mapping elements of "points_in" to "points_out", 1-based, Length: [ n_points_out ]
    //  3 - reverse_index   Indices for mapping elements of "points_out" to "points_in", 1-based, Length: [ n_points_in ]

    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:point_cloud_segmentation:IO_error", "Wrong number of input arguments.");

    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:point_cloud_segmentation:IO_error", "Too many output arguments.");

    arma::uword target_size = (nrhs < 2) ? 1024 : qd_mex_get_scalar<arma::uword>(prhs[1], "target_size", 1024);
    arma::uword vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "vec_size", 1);

    arma::fmat points_out_single;
    arma::mat points_out_double;
    arma::u32_vec sub_cloud_index, forward_index, reverse_index;

    if (mxIsSingle(prhs[0]))
    {
        const arma::fmat points_in = qd_mex_get_single_Mat(prhs[0]);
        CALL_QD(quadriga_lib::point_cloud_segmentation<float>(&points_in, &points_out_single,
                                                              &sub_cloud_index, target_size, vec_size,
                                                              &forward_index, &reverse_index));
    }
    else
    {
        const arma::mat points_in = qd_mex_get_double_Mat(prhs[0]);
        CALL_QD(quadriga_lib::point_cloud_segmentation<double>(&points_in, &points_out_double,
                                                              &sub_cloud_index, target_size, vec_size,
                                                              &forward_index, &reverse_index));
    }

    // Copy data to MATLAB / Octave
    if (nlhs > 0 && mxIsSingle(prhs[0]))
        plhs[0] = qd_mex_copy2matlab(&points_out_single);
    else if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&points_out_double);

    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&sub_cloud_index);

    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&forward_index);

    if (nlhs > 3)
    {
        reverse_index += 1;
        plhs[3] = qd_mex_copy2matlab(&reverse_index);
    }
}
