// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# POINT_CLOUD_AABB
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_index` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

## Usage:
```
aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_index, vec_size );
```

## Inputs:
- **`points`** — 3D point coordinates; `[n_points, 3]`
- **`sub_cloud_index`** *(optional)* — 1-based row indices marking the start of each sub-cloud;
  use [[point_cloud_segmentation]] to generate; uint32; `[n_sub]`
- **`vec_size`** *(optional)* — SIMD alignment padding factor (e.g. 4, 8, 16); default: 1

## Outputs:
- **`aabb`** — Bounding box matrix; `[n_out, 6]` where `n_out` is `n_sub` padded to a multiple of `vec_size`

## See also:
- [[point_cloud_segmentation]] (generate sub-cloud indices)
- [[ray_point_intersect]] (use AABBs for intersection)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const arma::mat points = qd_mex_get_Mat<double>(prhs[0]);
    arma::u32_vec sub_cloud_index = (nrhs < 2) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[1], true);
    const arma::uword vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "vec_size", 1);

    // Convert sub_mesh_index to 0-based
    for (unsigned &val : sub_cloud_index)
    {
        if (val == 0)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Values in 'sub_cloud_index' cannot be 0.");
        --val;
    }

    // Wrap optional input pointer
    const arma::u32_vec *p_sub_cloud_index = sub_cloud_index.empty() ? nullptr : &sub_cloud_index;

    // Output (returned by value, size determined at runtime)
    arma::mat aabb;

    // Call library function
    CALL_QD(aabb = quadriga_lib::point_cloud_aabb<double>(&points, p_sub_cloud_index, vec_size));

    // Copy to MATLAB
    if (nlhs > 0)
        plhs[0] = qd_mex_copy2matlab(&aabb);
}