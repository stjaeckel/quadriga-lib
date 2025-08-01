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
# point_cloud_segmentation
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
from quadriga_lib import RTtools

# Output as tuple
data = RTtools.point_cloud_segmentation( points, target_size, vec_size )

# Unpacked outputs
points_out, sub_cloud_ind, forward_ind, reverse_ind = RTtools.point_cloud_segmentation( points, target_size, vec_size )
```

## Input Arguments:
- **`points`**<br>
  Points in 3D-Cartesian space; Size: `[ n_points, 3 ]`

- **`target_size`** (optional)<br>
  The target number of elements of each sub-cloud. Default value = 1024. For best performance, the
  value should be around 10 * sgrt( n_points )

- **`vec_size`** (optional)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1.
  For values > 1,the number of rows for each sub-cloud in the output is increased to a multiple
  of `vec_size`. For padding, zero-sized triangles are placed at the center of the AABB of
  the corresponding sub-cloud.

## Output Arguments:
- **`points_out`**, `data[0]`<br>
  Points in 3D-Cartesian space; singe or double precision;  Size: `[ n_points_out, 3 ]`

- **`sub_cloud_ind`**, `data[1]`<br>
  Start indices of the sub-clouds in 0-based notation. Type: uint32; Vector of length `[ n_sub_cloud ]`

- **`forward_ind`**, `data[2]`<br>
  Indices for mapping elements of "points_in" to "points_out"; 1-based;
  Length: `[ n_points_out ]`; For `vec_size > 1`, the added elements not contained in the input
  are indicated by zeros.

- **`reverse_ind`**, `data[3]`<br>
  Indices for mapping elements of "points_out" to "points"; 0-based; Length: `[ n_points ]`
MD!*/

py::tuple point_cloud_segmentation(const py::array_t<double> &points, // Points in 3D-Cartesian space; Size: [ n_points, 3 ]
                                   arma::uword target_size,           // Target value for the sub-mesh size
                                   arma::uword vec_size)              // Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA)
{
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);

    arma::mat points_out;
    arma::u32_vec sub_cloud_ind, forward_ind, reverse_ind;

    quadriga_lib::point_cloud_segmentation<double>(&points_arma, &points_out, &sub_cloud_ind,
                                                   target_size, vec_size, &forward_ind, &reverse_ind);

    auto points_out_p = qd_python_copy2numpy(points_out);
    auto sub_cloud_ind_p = qd_python_copy2numpy<arma::u32, ssize_t>(sub_cloud_ind);
    auto forward_ind_p = qd_python_copy2numpy<arma::u32, ssize_t>(forward_ind);
    auto reverse_ind_p = qd_python_copy2numpy<arma::u32, ssize_t>(reverse_ind);

    return py::make_tuple(points_out_p, sub_cloud_ind_p, forward_ind_p, reverse_ind_p);
}
