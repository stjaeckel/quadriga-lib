// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_index` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

## Usage:
```
aabb = quadriga_lib.RTtools.point_cloud_aabb( points, sub_cloud_ind, vec_size )
```

## Input Arguments:
- **`points`** — 3D point coordinates; `(n_points, 3)`
- **`sub_cloud_index`** *(optional)* — 0-based row indices marking the start of each sub-cloud;
  use [[point_cloud_segmentation]] to generate; uint32; `(n_sub,)`
- **`vec_size`** *(optional)* — SIMD alignment padding factor (e.g. 4, 8, 16); default: 1

## Output Argument:
- **`aabb`** — Bounding box matrix; `(n_out, 6)` where `n_out` is `n_sub` padded to a multiple of `vec_size`

## See also:
- [[point_cloud_segmentation]] (generate sub-cloud indices)
- [[ray_point_intersect]] (use AABBs for intersection)
MD!*/

py::array_t<double> point_cloud_aabb(const py::array_t<double> &points,
                                     const py::array_t<unsigned> &sub_cloud_ind,
                                     arma::uword vec_size)

{
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);
    const auto sub_cloud_ind_arma = qd_python_numpy2arma_Col(sub_cloud_ind, true);

    arma::mat aabb = quadriga_lib::point_cloud_aabb(&points_arma, &sub_cloud_ind_arma, vec_size);

    return qd_python_copy2numpy(aabb);
}
