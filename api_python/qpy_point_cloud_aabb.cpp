// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2025 Stephan Jaeckel (https://sjc-wireless.com)
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

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# point_cloud_aabb
Calculates the axis-aligned bounding box (AABB) for a 3D point cloud or a set of sub-clouds

## Description:
The axis-aligned bounding box (AABB) of a set of points is the smallest bounding box whose edges
are parallel to the Cartesian coordinate axes and that fully contains all points in the set.
This function computes the AABB for either:
- an entire point cloud, or
- multiple sub-clouds defined by their starting indices.

For each cloud, the function returns the minimum and maximum extents along the x, y, and z
dimensions, producing a matrix where each row corresponds to one sub-cloud’s bounding box. When
SIMD-friendly alignment is required, the output is zero-padded to the nearest multiple of vec_size;
these padding rows should be ignored if the number of sub-clouds is known externally.<br><br>

If a sub_cloud_index is provided, the last index is assumed to extend to the end of the
points matrix. This functionality is particularly useful for preprocessing in geometry analysis,
rendering pipelines, and spatial acceleration structures such as BVHs or octrees. Sub-clouds
can be conveniently generated using [[point_cloud_segmentation]].

## Usage:
```
from quadriga_lib import RTtools

aabb = RTtools.point_cloud_aabb( points, sub_cloud_ind, vec_size )
```

## Input Arguments:
- **`points`**<br>
  Points in 3D-Cartesian space; Size: [ n_points, 3 ]

- **`sub_cloud_ind`** (optional)<br>
  Start indices of the sub-clouds in 0-based notation. If this parameter is not given, the AABB of
  the entire point cloud is returned. Vector of length `[ n_sub_cloud ]`

- **`vec_size`** (optional)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1. For values > 1,
  the number of rows in the output is increased to a multiple of `vec_size`, padded with zeros.

## Output Argument:
- **`aabb`**<br>
  Axis-aligned bounding box of each sub-cloud. Each box is described by 6 values:
  `[ x_min, x_max, y_min, y_max, z_min, z_max ]`; Size: `[ n_sub_cloud, 6 ]`

## See also:
- [[point_cloud_segmentation]] (for calculating sub clouds)
MD!*/

py::array_t<double> point_cloud_aabb(const py::array_t<double> &points,          // Points in 3D-Cartesian space; Size: [ n_points, 3 ]
                                     const py::array_t<unsigned> &sub_cloud_ind, // Start indices of the sub-clouds in 0-based notation
                                     arma::uword vec_size)                       // Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA)

{
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);
    const auto sub_cloud_ind_arma = qd_python_numpy2arma_Col(sub_cloud_ind, true);

    arma::mat aabb = quadriga_lib::point_cloud_aabb(&points_arma, &sub_cloud_ind_arma, vec_size);

    return qd_python_copy2numpy(aabb);
}
