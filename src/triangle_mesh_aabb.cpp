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

#include "quadriga_tools.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

## Description:
The axis-aligned minimum bounding box (or AABB) for a given set of triangles is its minimum
bounding box subject to the constraint that the edges of the box are parallel to the (Cartesian)
coordinate axes. Axis-aligned bounding boxes are used as an approximate location of the set of
triangles. In order to find intersections with the triangles (e.g. using ray tracing), the
initial check is the intersections between the rays and the AABBs. Since it is usually a much
less expensive operation than the check of the actual intersection (because it only requires
comparisons of coordinates), it allows quickly excluding checks of the pairs that are far apart.<br><br>

- This function computes the axis-aligned bounding box for each sub-mesh in a 3D triangle mesh.
- Each triangle is defined by three vertices in a flat row: `[x1, y1, z1, x2, y2, z2, x3, y3, z3]`.
- Sub-meshes are defined by the `sub_mesh_index` list, indicating the starting row of each sub-mesh.
- The resulting bounding boxes are returned as a matrix of shape `[n_sub, 6]` with columns: `[x_min, x_max, y_min, y_max, z_min, z_max]`.
- If `vec_size > 1`, the result is padded such that the number of rows in the output is a multiple of `vec_size`.
- Allowed datatypes (`dtype`): `float` or `double`.

## Declaration:
```
arma::Mat<dtype> quadriga_lib::triangle_mesh_aabb(
                const arma::Mat<dtype> *mesh,
                const arma::u32_vec *sub_mesh_index = nullptr,
                arma::uword vec_size = 1);
```

## Arguments:
- `const arma::Mat<dtype> ***mesh**` (input)<br>
  Vertices of the triangle mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[ n_triangles, 9 ]`

- `const arma::u32_vec ***sub_mesh_index** = nullptr` (optional input)<br>
  Start indices of the sub-meshes in 0-based notation. If this parameter is not given, the AABB of
  the entire triangle mesh is returned; Length `[n_sub]`

- `arma::uword **vec_size** = 1` (optional input)<br>
  Alignment size for SIMD processing (e.g., `8` for AVX2, `32` for CUDA). 
  Output is padded to a multiple of this value.

## Returns:
- `arma::Mat<dtype>`<br> 
  A matrix of shape `[n_sub_aligned, 6]`, where each row is `[x_min, x_max, y_min, y_max, z_min, z_max]`.
MD!*/

template <typename dtype>
arma::Mat<dtype> quadriga_lib::triangle_mesh_aabb(const arma::Mat<dtype> *mesh,
                                                  const arma::u32_vec *sub_mesh_index,
                                                  arma::uword vec_size)
{
    // Input validation
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mesh->n_rows == 0)
        throw std::invalid_argument("Input 'mesh' must have at least one face.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    arma::uword n_face = mesh->n_rows;
    arma::uword n_mesh = mesh->n_elem;
    const dtype *p_mesh = mesh->memptr();

    const unsigned first_sub_mesh_ind = 0;
    const unsigned *p_sub = &first_sub_mesh_ind;
    arma::uword n_sub = 1;

    if (sub_mesh_index != nullptr && sub_mesh_index->n_elem > 0)
    {
        n_sub = sub_mesh_index->n_elem;
        if (n_sub == 0)
            throw std::invalid_argument("Input 'sub_mesh_index' must have at least one element.");

        p_sub = sub_mesh_index->memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-mesh must start at index 0.");

        for (arma::uword i = 1; i < n_sub; ++i)
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-mesh indices must be sorted in ascending order.");

        if (p_sub[n_sub - 1] >= (unsigned)n_face)
            throw std::invalid_argument("Sub-mesh indices cannot exceed number of faces.");
    }

    // Reserve memory for the output
    arma::uword n_out = (n_sub % vec_size == 0) ? n_sub : (n_sub / vec_size + 1) * vec_size;
    arma::Mat<dtype> output(n_out, 6); // Initialized to 0

    dtype *x_min = output.colptr(0), *x_max = output.colptr(1),
          *y_min = output.colptr(2), *y_max = output.colptr(3),
          *z_min = output.colptr(4), *z_max = output.colptr(5);

    for (arma::uword i = 0; i < n_sub; ++i)
    {
        x_min[i] = INFINITY, x_max[i] = -INFINITY,
        y_min[i] = INFINITY, y_max[i] = -INFINITY,
        z_min[i] = INFINITY, z_max[i] = -INFINITY;
    }

    arma::uword i_sub = 0, i_next = n_face;
    for (arma::uword i_mesh = 0; i_mesh < n_mesh; ++i_mesh)
    {
        dtype v = p_mesh[i_mesh];            // Mesh value
        arma::uword i_col = i_mesh / n_face; // Column index in mesh
        arma::uword i_row = i_mesh % n_face; // Row index in mesh

        if (i_row == 0)
        {
            i_sub = 0;
            i_next = (i_sub == n_sub - 1) ? n_face : (arma::uword)p_sub[i_sub + 1];
        }
        else if (i_row == i_next)
        {
            ++i_sub;
            i_next = (i_sub == n_sub - 1) ? n_face : (arma::uword)p_sub[i_sub + 1];
        }

        if (i_col % 3 == 0)
        {
            x_min[i_sub] = (v < x_min[i_sub]) ? v : x_min[i_sub];
            x_max[i_sub] = (v > x_max[i_sub]) ? v : x_max[i_sub];
        }
        else if (i_col % 3 == 1)
        {
            y_min[i_sub] = (v < y_min[i_sub]) ? v : y_min[i_sub];
            y_max[i_sub] = (v > y_max[i_sub]) ? v : y_max[i_sub];
        }
        else
        {
            z_min[i_sub] = (v < z_min[i_sub]) ? v : z_min[i_sub];
            z_max[i_sub] = (v > z_max[i_sub]) ? v : z_max[i_sub];
        }
    }

    return output;
}

template arma::Mat<float> quadriga_lib::triangle_mesh_aabb(const arma::Mat<float> *mesh,
                                                           const arma::u32_vec *sub_mesh_index,
                                                           arma::uword vec_size);

template arma::Mat<double> quadriga_lib::triangle_mesh_aabb(const arma::Mat<double> *mesh,
                                                            const arma::u32_vec *sub_mesh_index,
                                                            arma::uword vec_size);
