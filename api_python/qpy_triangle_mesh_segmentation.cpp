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
# TRIANGLE_MESH_SEGMENTATION
Rearranges elements of a triangle mesh into smaller sub-meshes

## Description:
This function processes the elements of a large triangle mesh by clustering those that are
closely spaced. The resulting mesh retains the same elements but rearranges their order.
The function aims to minimize the size of the axis-aligned bounding box around each cluster,
referred to as a sub-mesh, while striving to maintain a specific number of elements within
each cluster.<br><br>

This approach is particularly useful in computer graphics and simulation applications where
managing computational resources efficiently is crucial. By organizing the mesh elements into
compact clusters, the function enhances rendering performance and accelerates computational
tasks, such as collision detection and physics simulations. It allows for quicker processing
and reduced memory usage, making it an essential technique in both real-time graphics rendering
and complex simulation environments.

## Usage:
```
from quadriga_lib import RTtools

# Output as tuple
data = RTtools.triangle_mesh_segmentation( triangles, target_size, vec_size, mtl_prop )

# Unpacked outputs
triangles_out, sub_mesh_index, mesh_index, mtl_prop_out = RTtools.triangle_mesh_segmentation( triangles, target_size, vec_size, mtl_prop )
```

## Input Arguments:
- **`triangles`**<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[ n_triangles, 9 ]`

- **`target_size`** (optional)<br>
  The target number of elements of each sub-mesh. Default value = 1024. For best performance, the
  value should be around `sgrt( n_triangles )`

- **`vec_size`** (optional)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1.
  For values > 1,the number of rows for each sub-mesh in the output is increased to a multiple
  of `vec_size`. For padding, zero-sized triangles are placed at the center of the AABB of
  the corresponding sub-mesh.

- **`mtl_prop_in`** (optional)<br>
  Material properties of each mesh element; Size: `[ n_triangles, 5 ]`
  If this is not provided, the corresponding `mtl_prop_out` will be empty.

## Output Arguments:
- **`triangles_out`**, `data[0]`<br>
  Vertices of the clustered mesh in global Cartesian coordinates; Size: `[ n_triangles_out, 9 ]`

- **`sub_mesh_index`**, `data[1]`<br>
  Start indices of the sub-meshes in 0-based notation. Type: int; Vector of length `[ n_sub_mesh ]`

- **`mesh_index`**, `data[2]`<br>
  Indices for mapping elements of "triangles_in" to "triangles_out"; 1-based;
  Length: `[ n_triangles_out ]`; For `vec_size > 1`, the added elements not contained in the input
  are indicated by zeros.

- **`mtl_prop_out`**, `data[3]`<br>
  Material properties for the sub-divided triangle mesh elements. The values for the new faces are
  copied from `mtl_prop_in`; Size: `[ n_triangles_out, 5 ]`; For `vec_size > 1`, the added elements
  will contain the vacuum / air material.
MD!*/

py::tuple triangle_mesh_segmentation(const py::array_t<double> &triangles, // Faces of the triangular mesh (input), Size: [ n_mesh, 9 ]
                                     arma::uword target_size,              // Target value for the sub-mesh size
                                     arma::uword vec_size,                 // Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA)
                                     const py::array_t<double> &mtl_prop)  // Material properties (input), Size: [ n_mesh, 5 ], optional
{
    const auto triangles_arma = qd_python_numpy2arma_Mat(triangles, true);
    const auto mtl_prop_arma = qd_python_numpy2arma_Mat(mtl_prop, true);

    arma::mat triangles_out_arma, mtl_prop_out_arma;
    arma::u32_vec sub_mesh_index, mesh_index;

    quadriga_lib::triangle_mesh_segmentation(&triangles_arma, &triangles_out_arma,
                                             &sub_mesh_index, target_size, vec_size,
                                             &mtl_prop_arma, &mtl_prop_out_arma, &mesh_index);

    auto triangles_p = qd_python_copy2numpy(triangles_out_arma);
    auto sub_mesh_index_p = qd_python_copy2numpy<arma::u32, ssize_t>(sub_mesh_index);
    auto mesh_index_p = qd_python_copy2numpy<arma::u32, ssize_t>(mesh_index);
    auto mtl_prop_p = qd_python_copy2numpy(mtl_prop_out_arma);

    return py::make_tuple(triangles_p, sub_mesh_index_p, mesh_index_p, mtl_prop_p);
}