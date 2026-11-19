// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

## Declaration:
```
arma::Mat<dtype> quadriga_lib::triangle_mesh_aabb(
    const arma::Mat<dtype> &mesh,
    const arma::u32_vec &sub_mesh_ind = arma::u32_vec(),
    arma::uword vec_size = 1);
```

## Inputs:
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; further columns are ignored; `[n_triangles, 9]`
- **`sub_mesh_ind`** *(optional)* — 0-based start indices of sub-meshes; if omitted, the AABB of the entire mesh is returned; `[n_sub]`
- **`vec_size`** *(optional)* — Alignment size for SIMD/CUDA padding (e.g., `8` for AVX2, `32` for CUDA)

## Returns:
- Bounding boxes `aabb`, one row per sub-mesh; `[n_out, 6]` where `n_out` is `n_sub` padded to a multiple of `vec_size`

## See also:
- [[ray_triangle_intersect]] (consumer of the output)
MD!*/

template <typename dtype>
arma::Mat<dtype> quadriga_lib::triangle_mesh_aabb(const arma::Mat<dtype> &mesh,
                                                  const arma::u32_vec &sub_mesh_ind,
                                                  arma::uword vec_size)
{
    // Input validation
    if (mesh.n_cols < 9)
        throw std::invalid_argument("Input 'mesh' must have at least 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mesh.n_rows == 0)
        throw std::invalid_argument("Input 'mesh' must have at least one face.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    arma::uword n_face = mesh.n_rows;
    arma::uword n_mesh = 9 * n_face; // First 9 columns (column-major), further columns are ignored
    const dtype *p_mesh = mesh.memptr();

    const unsigned first_sub_mesh_ind = 0;
    const unsigned *p_sub = &first_sub_mesh_ind;
    arma::uword n_sub = 1;

    if (sub_mesh_ind.n_elem > 0)
    {
        n_sub = sub_mesh_ind.n_elem;
        p_sub = sub_mesh_ind.memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-mesh must start at the first face.");

        for (arma::uword i = 1; i < n_sub; ++i)
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-mesh indices must be sorted in ascending order.");

        if (p_sub[n_sub - 1] >= (unsigned)n_face)
            throw std::invalid_argument("Sub-mesh indices cannot exceed number of faces.");
    }

    // Padding rows stay zero
    arma::uword n_out = (n_sub % vec_size == 0) ? n_sub : (n_sub / vec_size + 1) * vec_size;
    arma::Mat<dtype> aabb(n_out, 6, arma::fill::zeros);

    dtype *x_min = aabb.colptr(0), *x_max = aabb.colptr(1),
          *y_min = aabb.colptr(2), *y_max = aabb.colptr(3),
          *z_min = aabb.colptr(4), *z_max = aabb.colptr(5);

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
    return aabb;
}

template arma::Mat<float> quadriga_lib::triangle_mesh_aabb(const arma::Mat<float> &mesh,
                                                           const arma::u32_vec &sub_mesh_ind,
                                                           arma::uword vec_size);

template arma::Mat<double> quadriga_lib::triangle_mesh_aabb(const arma::Mat<double> &mesh,
                                                            const arma::u32_vec &sub_mesh_ind,
                                                            arma::uword vec_size);
