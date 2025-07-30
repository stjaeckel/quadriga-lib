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
# RAY_TRIANGLE_INTERSECT
Calculates the intersection of rays and triangles in three dimensions

## Description:
- This function implements the Möller–Trumbore ray-triangle intersection algorithm, known for its
  efficiency in calculating the intersection of a ray and a triangle in three-dimensional space.
  This method achieves its speed by eliminating the need for precomputed plane equations of the plane
  containing the triangle.

- For further information, refer to [Wikipedia: <a href="https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm">Möller–Trumbore intersection algorithm</a>].

- The algorithm defines the ray using two points: an origin and a destination. Similarly, the triangle
  is specified by its three vertices.

- To enhance performance, this implementation leverages AVX2 intrinsic functions and OpenMP, when
  available, to speed up the computational process.

## Usage:
```
from quadriga_lib import RTtools

# Output as tuple
data = RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_index )

# Unpacked outputs
fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_index )
```

## Input Arguments:
- **`orig`**<br>
  Ray origins in 3D Cartesian coordinates; Size: `[ no_ray, 3 ]`

- **`dest`**<br>
  Ray destinations in 3D Cartesian coordinates; Size: `[ no_ray, 3 ]`

- **`mesh`**<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3 points
  in 3D-space. Hence, a face has 9 values in the order [ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]; 
  Size: `[ no_mesh, 9 ]`

- **`sub_mesh_index`** (optional)<br>
  Start indices of the sub-meshes in 0-based notation. If this parameter is not given, intersections
  are calculated for each mesh element, leading to poor performance for large meshes.
  Vector of length `[ n_sub_mesh ]`

## Output Arguments:
- **`fbs`**, `data[0]`<br>
  First interaction point between the rays and the triangular mesh. If no interaction was found, the
  FBS location is equal to `dest`. Size: `[ no_ray, 3 ]`

- **`sbs`**, `data[1]`<br>
  Second interaction point between the rays and the triangular mesh. If no interaction was found, the
  SBS location is equal to `dest`. Size: `[ no_ray, 3 ]`

- **`no_interact`**, `data[2]`<br>
  Total number of interactions between the origin point and the destination; uint32; Length: `[ no_ray ]`

- **`fbs_ind`**, `data[3]`<br>
  Index of the triangle that was hit by the ray at the FBS location; 1-based; uint32; Length: `[ no_ray ]`

- **`sbs_ind`**, `data[4]`<br>
  Index of the triangle that was hit by the ray at the SBS location; 1-based; uint32; Length: `[ no_ray ]`

## Caveat:
- All internal computation are done in single precision to achieve an additional 2x improvement in
  speed compared to double precision when using AVX2 intrinsic instructions

## See also:
- [[obj_file_read]] (for loading mesh from an OBJ file)
- [[icosphere]] (for generating beams)
- [[triangle_mesh_segmentation]] (for calculating sub-meshes)
MD!*/

py::tuple ray_triangle_intersect(const py::array_t<double> &orig,             // Ray origin points in GCS, Size [ n_ray, 3 ]
                                 const py::array_t<double> &dest,             // Ray destination points in GCS, Size [ n_ray, 3 ]
                                 const py::array_t<double> &mesh,             // Faces of the triangular mesh (input), Size: [ n_mesh, 9 ]
                                 const py::array_t<unsigned> &sub_mesh_index) // Start indices of the sub-meshes in 0-based notation
{
    const auto orig_arma = qd_python_numpy2arma_Mat(orig, true);
    const auto dest_arma = qd_python_numpy2arma_Mat(dest, true);
    const auto triangles_arma = qd_python_numpy2arma_Mat(mesh, true);
    const auto sub_mesh_index_arma = qd_python_numpy2arma_Col(sub_mesh_index, true);

    arma::mat fbs, sbs;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;

    quadriga_lib::ray_triangle_intersect<double>(&orig_arma, &dest_arma, &triangles_arma,
                                                 &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind,
                                                 &sub_mesh_index_arma);

    auto fbs_p = qd_python_copy2numpy(fbs);
    auto sbs_p = qd_python_copy2numpy(sbs);
    auto no_interact_p = qd_python_copy2numpy<arma::u32, ssize_t>(no_interact);
    auto fbs_ind_p = qd_python_copy2numpy<arma::u32, ssize_t>(fbs_ind);
    auto sbs_ind_p = qd_python_copy2numpy<arma::u32, ssize_t>(sbs_ind);

    return py::make_tuple(fbs_p, sbs_p, no_interact_p, fbs_ind_p, sbs_ind_p);
}
