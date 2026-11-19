// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "qd_helpers_mex.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# POINT_CLOUD_SEGMENTATION
Reorganize a point cloud into spatial sub-clouds for efficient processing

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes
  at the midpoint
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are
  placed at the sub-cloud AABB center
- Produces a reorganized point array and index maps to track reordering

## Usage:
```
[ points_out, sub_cloud_ind, forward_ind, reverse_ind ] = ...
    quadriga_lib.point_cloud_segmentation( points, target_size, vec_size );
```

## Inputs:
- **`points`** — Original 3D point cloud; `[n_points, 3]`
- **`target_size`** — Maximum points per sub-cloud before padding; default: 1024
- **`vec_size`** — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of this value; no padding when `1`;
  [[ray_point_intersect]] accepts any value, padding (e.g. 8 for AVX2) only improves its speed; default: 1

## Outputs:
- **`points_out`** — Reorganized point cloud with points grouped by sub-cloud; `[n_points_out, 3]`
- **`sub_cloud_ind`** — 1-based starting index of each sub-cloud within `points_out`; `[n_sub]`
- **`forward_ind`** — 1-based row in `points` of each row of `points_out`; padding entries are `0`; `[n_points_out]`
- **`reverse_ind`** — 1-based row in `points_out` of each row of `points`; `[n_points]`
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 1 || nrhs > 3)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 4)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input arguments
    const arma::mat points = qd_mex_get_Mat<double>(prhs[0], false, "points");
    const arma::uword target_size = (nrhs < 2) ? 1024 : qd_mex_get_scalar<arma::uword>(prhs[1], "target_size", 1024);
    const arma::uword vec_size = (nrhs < 3) ? 1 : qd_mex_get_scalar<arma::uword>(prhs[2], "vec_size", 1);

    // Outputs (sizes are determined by C++ at runtime)
    arma::mat points_out;
    arma::u32_vec sub_cloud_ind, forward_ind, reverse_ind;

    // Wrap optional output pointers using nlhs guard
    // (.empty() guard is unsafe here because C++ resizes these outputs)
    arma::u32_vec *p_forward_index = (nlhs > 2) ? &forward_ind : nullptr;
    arma::u32_vec *p_reverse_index = (nlhs > 3) ? &reverse_ind : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::point_cloud_segmentation<double>(points, points_out, sub_cloud_ind,
                                                           target_size, vec_size,
                                                           p_forward_index, p_reverse_index));

    // Convert 0-based C++ indices to 1-based MATLAB indices
    // (forward_ind is already 1-based in C++ with 0 for padding — no conversion)
    sub_cloud_ind += 1;
    if (p_reverse_index != nullptr)
        reverse_ind += 1;

    // Copy outputs to MATLAB
    plhs[0] = qd_mex_copy2matlab(&points_out);
    if (nlhs > 1)
        plhs[1] = qd_mex_copy2matlab(&sub_cloud_ind);
    if (nlhs > 2)
        plhs[2] = qd_mex_copy2matlab(&forward_ind);
    if (nlhs > 3)
        plhs[3] = qd_mex_copy2matlab(&reverse_ind);
}