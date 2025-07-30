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
# TRIANGLE_MESH_AABB
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

## Description:
The axis-aligned minimum bounding box (or AABB) for a given set of triangles is its minimum
bounding box subject to the constraint that the edges of the box are parallel to the (Cartesian)
coordinate axes. Axis-aligned bounding boxes are used as an approximate location of the set of
triangles. In order to find intersections with the triangles (e.g. using ray tracing), the
initial check is the intersections between the rays and the AABBs. Since it is usually a much
less expensive operation than the check of the actual intersection (because it only requires
comparisons of coordinates), it allows quickly excluding checks of the pairs that are far apart.

## Usage:
```
from quadriga_lib import RTtools

aabb = RTtools.triangle_mesh_aabb( triangle_mesh, sub_mesh_index, vec_size );
```

## Input Arguments:
- **`triangle_mesh`**<br>
  Vertices of the triangle mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[ n_triangles, 9 ]`

- **`sub_mesh_index`** (optional)<br>
  Start indices of the sub-meshes in 0-based notation. If this parameter is not given, the AABB of
  the entire triangle mesh is returned. Vector of length `[ n_sub_mesh ]`

- **`vec_size`** (optional)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1. For values > 1,
  the number of rows in the output is increased to a multiple of `vec_size`, padded with zeros.

## Output Argument:
- **`aabb`**<br>
  Axis-aligned bounding box of each sub-mesh. Each box is described by 6 values:
  `[ x_min, x_max, y_min, y_max, z_min, z_max ]`; Size: `[ n_sub_mesh, 6 ]`

## See also:
- [[triangle_mesh_segmentation]] (for calculating sub-meshes)
MD!*/

py::array_t<double> triangle_mesh_aabb(const py::array_t<double> &triangles,        // Faces of the triangular mesh (input), Size: [ n_mesh, 9 ]
                                       const py::array_t<unsigned> &sub_mesh_index, // Start indices of the sub-meshes in 0-based notation
                                       arma::uword vec_size)                        // Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA)
{
    const auto triangles_arma = qd_python_numpy2arma_Mat(triangles, true);
    const auto sub_mesh_index_arma = qd_python_numpy2arma_Col(sub_mesh_index, true);

    arma::mat aabb = quadriga_lib::triangle_mesh_aabb(&triangles_arma, &sub_mesh_index_arma, vec_size);

    return qd_python_copy2numpy(aabb);
}
