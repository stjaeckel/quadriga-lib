// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes at the midpoint
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are placed at the sub-cloud AABB center
- Produces a reorganized point array and index maps to track reordering

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.point_cloud_segmentation( points, target_size, vec_size )

# Unpacked outputs
points_out, sub_cloud_ind, forward_ind, reverse_ind =
    quadriga_lib.RTtools.point_cloud_segmentation( points, target_size, vec_size )
```

## Inputs:
- **`points`** — Original 3D point cloud; `(n_points, 3)`
- **`target_size`** *(optional)* — Maximum points per sub-cloud before padding; default: 1024
- **`vec_size`** *(optional)* — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of
  this value; no padding when `1`; default: 1

## Outputs:
- **`points_out`** — Reorganized point cloud with points grouped by sub-cloud; `(n_points_out, 3)`
- **`sub_cloud_index`** — 0-based starting index of each sub-cloud within `points_out`; uint32; `(n_sub,)`
- **`forward_index`** *(optional)* — 1-based index map from `points` to `points_out`; padding entries are `0`; uint32; `(n_points_out,)`
- **`reverse_index`** *(optional)* — 0-based index map from `points_out` back to `points`; uint32; `(n_points,)`
MD!*/

py::tuple point_cloud_segmentation(const py::array_t<double> &points,
                                   arma::uword target_size,
                                   arma::uword vec_size)
{
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);

    arma::mat points_out;
    arma::u32_vec sub_cloud_ind, forward_ind, reverse_ind;

    quadriga_lib::point_cloud_segmentation<double>(&points_arma, &points_out, &sub_cloud_ind,
                                                   target_size, vec_size, &forward_ind, &reverse_ind);

    auto points_out_p = qd_python_copy2numpy(&points_out);
    auto sub_cloud_ind_p = qd_python_copy2numpy(&sub_cloud_ind);
    auto forward_ind_p = qd_python_copy2numpy(&forward_ind);
    auto reverse_ind_p = qd_python_copy2numpy(&reverse_ind);

    return py::make_tuple(points_out_p, sub_cloud_ind_p, forward_ind_p, reverse_ind_p);
}
