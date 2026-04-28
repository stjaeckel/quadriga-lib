// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# triangle_mesh_split
Split a 3D triangular mesh into two sub-meshes along a given axis

- Splits at the bounding box center of the selected axis; triangles where all vertices lie within the 
  lower half go to `meshA`; any triangle with at least one vertex exceeding the threshold goes to `meshB`
- `axis = 0` selects the axis with the longest bounding box extent automatically
- On failure (all triangles fall to one side), `meshA` and `meshB` are left unchanged and the return value is negative
- Used internally by [[triangle_mesh_segmentation]]

## Declaration:
```
int triangle_mesh_split(
    const arma::Mat<dtype> *mesh,
    arma::Mat<dtype> *meshA,
    arma::Mat<dtype> *meshB,
    int axis = 0,
    arma::Col<int> *split_ind = nullptr);
```

## Inputs:
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`axis`** *(optional)* — Split axis: 0 = longest extent, 1 = x, 2 = y, 3 = z

## Outputs:
- **`meshA`** — Triangles with all vertices within the lower half of the bounding box; `[n_meshA, 9]`
- **`meshB`** — Triangles with at least one vertex exceeding the split threshold; `[n_meshB, 9]`
- **`split_ind`** *(optional)* — Per-triangle assignment: 1 = meshA, 2 = meshB, 0 = unassigned (failure); `[n_mesh]`

## Returns:
- Axis used for the split (1, 2, or 3); negative (-1, -2, or -3) on failure

## See also:
- [[triangle_mesh_segmentation]] (calls this function recursively)
MD!*/

template <typename dtype>
int quadriga_lib::triangle_mesh_split(const arma::Mat<dtype> *mesh,
                                      arma::Mat<dtype> *meshA,
                                      arma::Mat<dtype> *meshB,
                                      int axis,
                                      arma::Col<int> *split_ind)
{
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (meshA == nullptr)
        throw std::invalid_argument("Output 'meshA' cannot be NULL.");
    if (meshB == nullptr)
        throw std::invalid_argument("Output 'meshB' cannot be NULL.");

    // Calculate bounding box
    arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb(mesh);

    arma::uword n_face = mesh->n_rows;
    arma::uword n_mesh = mesh->n_elem;
    const dtype *p_mesh = mesh->memptr();

    // Find longest axis
    dtype x = aabb.at(0, 1) - aabb.at(0, 0);
    dtype y = aabb.at(0, 3) - aabb.at(0, 2);
    dtype z = aabb.at(0, 5) - aabb.at(0, 4);

    if (axis == 0)
    {
        if (z >= y && z >= x)
            axis = 3;
        else if (y >= x && y >= z)
            axis = 2;
        else
            axis = 1;
    }
    else if (axis != 1 && axis != 2 && axis != 3)
        throw std::invalid_argument("Input 'axis' must have values 0, 1, 2 or 3.");

    // Define bounding box A
    dtype x_max = (axis == 1) ? aabb.at(0, 0) + (dtype)0.5 * x : aabb.at(0, 1),
          y_max = (axis == 2) ? aabb.at(0, 2) + (dtype)0.5 * y : aabb.at(0, 3),
          z_max = (axis == 3) ? aabb.at(0, 4) + (dtype)0.5 * z : aabb.at(0, 5);

    // Determine all mesh elements that are outside box A
    bool *isB = new bool[n_face](); // Init to false

    for (arma::uword i_mesh = 0; i_mesh < n_mesh; ++i_mesh)
    {
        dtype v = p_mesh[i_mesh];            // Mesh value
        arma::uword i_col = i_mesh / n_face; // Column index in mesh
        arma::uword i_row = i_mesh % n_face; // Row index in mesh

        if (i_col % 3 == 0)
            isB[i_row] = (v > x_max) ? true : isB[i_row];
        else if (i_col % 3 == 1)
            isB[i_row] = (v > y_max) ? true : isB[i_row];
        else
            isB[i_row] = (v > z_max) ? true : isB[i_row];
    }

    // Count items in both sub-meshes
    arma::uword n_faceA = 0, n_faceB = 0;
    for (arma::uword i = 0; i < n_face; ++i)
        if (isB[i])
            ++n_faceB;
        else
            ++n_faceA;

    // Check if the mesh was split
    if (n_faceA == 0 || n_faceB == 0)
    {
        delete[] isB;
        return -axis;
    }

    // Adjust output size
    meshA->set_size(n_faceA, 9);
    meshB->set_size(n_faceB, 9);

    dtype *p_meshA = meshA->memptr();
    dtype *p_meshB = meshB->memptr();

    bool write_split_ind = false;
    int *p_split_ind = nullptr;
    if (split_ind != nullptr)
    {
        write_split_ind = true;
        if (split_ind->n_elem != mesh->n_rows)
            split_ind->zeros(mesh->n_rows);
        else
            split_ind->zeros();
        p_split_ind = split_ind->memptr();
    }

    // Copy data
    arma::uword i_meshA = 0, i_meshB = 0;
    for (arma::uword i_mesh = 0; i_mesh < n_mesh; ++i_mesh)
    {
        dtype v = p_mesh[i_mesh];            // Mesh value
        arma::uword i_row = i_mesh % n_face; // Row index in mesh

        if (isB[i_row])
        {
            p_meshB[i_meshB++] = v;
            if (write_split_ind)
                p_split_ind[i_row] = 2;
        }
        else
        {
            p_meshA[i_meshA++] = v;
            if (write_split_ind)
                p_split_ind[i_row] = 1;
        }
    }

    delete[] isB;
    return axis;
}

template int quadriga_lib::triangle_mesh_split(const arma::Mat<float> *mesh,
                                               arma::Mat<float> *meshA,
                                               arma::Mat<float> *meshB,
                                               int axis,
                                               arma::Col<int> *split_ind);

template int quadriga_lib::triangle_mesh_split(const arma::Mat<double> *mesh,
                                               arma::Mat<double> *meshA,
                                               arma::Mat<double> *meshB,
                                               int axis,
                                               arma::Col<int> *split_ind);
