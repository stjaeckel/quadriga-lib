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
# triangle_mesh_segmentation
Reorganize a 3D mesh into smaller sub-meshes for faster processing

## Description:
This function processes the elements of a large triangle mesh by clustering those that are
closely spaced. The resulting mesh retains the same elements but rearranges their order.
The function aims to minimize the size of the axis-aligned bounding box around each cluster,
referred to as a sub-mesh, while striving to maintain a specific number of elements within
each cluster.<br><br>

- Subdivision is recursive and based on bounding box partitioning until each sub-mesh contains no more than `target_size` triangles.
- Sub-meshes are aligned to `vec_size` for SIMD or GPU optimization; padded with dummy triangles at the center of each sub-mesh if needed.
- If material properties are provided, these are also reorganized and padded accordingly.
- The function returns the number of created sub-meshes, and reorders the triangles and materials.
- Allowed datatypes (`dtype`): `float` or `double`.

## Declaration:
```
arma::uword quadriga_lib::triangle_mesh_segmentation(
                const arma::Mat<dtype> *mesh,
                arma::Mat<dtype> *meshR,
                arma::u32_vec *sub_mesh_index,
                arma::uword target_size = 1024,
                arma::uword vec_size = 1,
                const arma::Mat<dtype> *mtl_prop = nullptr,
                arma::Mat<dtype> *mtl_propR = nullptr,
                arma::u32_vec *mesh_index = nullptr);
```

## Arguments:
- `const arma::Mat<dtype> ***mesh**` (input)<br>
  Vertices of the triangular mesh in global Cartesian coordinates. Each face is described by 3
  points in 3D-space: `[ v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z ]`; Size: `[n_mesh, 9]`

- `arma::Mat<dtype> ***meshR**` (output)<br>
  Vertices of the clustered mesh in global Cartesian coordinates; Size: `[n_meshR, 9]`

- `arma::u32_vec ***sub_mesh_index**` (output)<br>
  Start indices of the sub-meshes in 0-based notation; Vector of length `[n_sub]`

- `arma::uword **target_size** = 1024` (input)<br>
  The target number of elements of each sub-mesh. Default value = 1024. For best performance, the
  value should be around `sgrt(n_mesh)`

- `arma::uword **vec_size** = 1` (input)<br>
  Vector size for SIMD processing (e.g. 8 for AVX2, 32 for CUDA). Default value = 1.
  For values > 1,the number of rows for each sub-mesh in the output is increased to a multiple
  of `vec_size`. For padding, zero-sized triangles are placed at the center of the AABB of
  the corresponding sub-mesh.

- `const arma::Mat<dtype> ***mtl_prop** = nullptr` (optional input)<br>
  Material properties corresponding to the original mesh; Size: `[n_mesh, 5]`; See <a href="#obj_file_read">obj_file_read</a>

- `arma::Mat<dtype> ***mtl_propR** = nullptr` (optional output)<br>
  Reorganized material properties, aligned and padded if necessary, Size: `[n_meshR, 5]`

- `arma::u32_vec ***mesh_index** = nullptr` (optional output)<br>
  1-based mapping from the original mesh to the reorganized mesh; Size: `[n_meshR]` (0 = padding)

## Returns:
- `arma::uword **n_sub**`<br>
  The number of created sub-meshes. Output matrices are resized accordingly.
MD!*/

template <typename dtype>
arma::uword quadriga_lib::triangle_mesh_segmentation(const arma::Mat<dtype> *mesh, arma::Mat<dtype> *meshR,
                                                     arma::u32_vec *sub_mesh_index, arma::uword target_size, arma::uword vec_size,
                                                     const arma::Mat<dtype> *mtl_prop, arma::Mat<dtype> *mtl_propR, arma::u32_vec *mesh_index)
{
    // Input validation
    if (mesh == nullptr)
        throw std::invalid_argument("Input 'mesh' cannot be NULL.");
    if (mesh->n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mesh->n_rows == 0)
        throw std::invalid_argument("Input 'mesh' must have at least one face.");

    arma::uword n_mesh = mesh->n_rows;

    if (meshR == nullptr)
        throw std::invalid_argument("Output 'meshR' cannot be NULL.");
    if (sub_mesh_index == nullptr)
        throw std::invalid_argument("Output 'sub_mesh_index' cannot be NULL.");

    if (target_size == 0)
        throw std::invalid_argument("Input 'target_size' cannot be 0.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    bool process_mtl_prop = (mtl_prop != nullptr) && (mtl_propR != nullptr) && (mtl_prop->n_elem != 0);

    if (process_mtl_prop)
    {
        if (mtl_prop->n_cols != 5)
            throw std::invalid_argument("Input 'mtl_prop' must have 5 columns.");

        if (mtl_prop->n_rows != mesh->n_rows)
            throw std::invalid_argument("Number of rows in 'mesh' and 'mtl_prop' dont match.");
    }

    // Create a vector of meshes
    std::vector<arma::Mat<dtype>> c; // Vector of sub-meshes

    // Add first mesh (creates a copy of the data)
    c.push_back(*mesh);

    // Create base index (0-based)
    std::vector<arma::uvec> face_ind; // Index list
    {
        arma::uvec base_index(mesh->n_rows, arma::fill::none);
        arma::uword *p = base_index.memptr();
        for (arma::uword i = 0; i < n_mesh; ++i)
            p[i] = (arma::uword)i;
        face_ind.push_back(std::move(base_index));
    }

    // Iterate through all elements
    for (auto sub_mesh_it = c.begin(); sub_mesh_it != c.end();)
    {
        arma::uword n_sub_faces = (*sub_mesh_it).n_rows;
        if (n_sub_faces > target_size)
        {
            arma::Mat<dtype> meshA, meshB;
            arma::s32_vec split_ind;

            // Split the mesh on its longest axis
            int split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 0, &split_ind);

            // Check the split proportions, must have at least a 10 / 90 split
            float p = (split_success > 0) ? (float)meshA.n_rows / (float)n_sub_faces : 0.5f;
            split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

            // Attempt to split along another axis if failed for longest axis
            int first_test = std::abs(split_success);
            if (split_success < 0) // Failed condition
            {
                // Test second axis
                if (first_test == 2 || first_test == 3) // Test x-axis
                    split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 1, &split_ind);
                else // Test y-axis
                    split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 2, &split_ind);

                p = (split_success > 0) ? (float)meshA.n_rows / (float)n_sub_faces : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

                // If we still failed, we test the third axis
                if ((first_test == 1 && split_success == -2) || (first_test == 2 && split_success == -1))
                    split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 3, &split_ind);
                else if (first_test == 3 && split_success == -1)
                    split_success = quadriga_lib::triangle_mesh_split(&(*sub_mesh_it), &meshA, &meshB, 2, &split_ind);

                p = (split_success > 0) ? (float)meshA.n_rows / (float)n_sub_faces : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;
            }

            // Update the mesh data in memory
            int *ps = split_ind.memptr();
            if (split_success > 0)
            {
                // Split the index list
                arma::uword i_sub = sub_mesh_it - c.begin(); // Sub-mesh index
                auto face_ind_it = face_ind.begin() + i_sub; // Face index iterator
                arma::uword *pi = (*face_ind_it).memptr();   // Current face index list

                arma::uvec meshA_index(meshA.n_rows, arma::fill::none);
                arma::uvec meshB_index(meshB.n_rows, arma::fill::none);
                arma::uword *pA = meshA_index.memptr(); // New face index list of mesh A
                arma::uword *pB = meshB_index.memptr(); // New face index list of mesh B

                arma::uword iA = 0, iB = 0;
                for (arma::uword i_face = 0; i_face < n_sub_faces; ++i_face)
                {
                    if (ps[i_face] == 1)
                        pA[iA++] = pi[i_face];
                    else if (ps[i_face] == 2)
                        pB[iB++] = pi[i_face];
                }

                face_ind.erase(face_ind_it);
                face_ind.push_back(std::move(meshA_index));
                face_ind.push_back(std::move(meshB_index));

                // Update the vector of sub-meshes
                c.erase(sub_mesh_it);
                c.push_back(std::move(meshA));
                c.push_back(std::move(meshB));
                sub_mesh_it = c.begin();
            }
            else
                ++sub_mesh_it;
        }
        else
            ++sub_mesh_it;
    }

    // Get the sub-mesh indices
    arma::uword n_sub = c.size(), n_out = 0;
    sub_mesh_index->set_size(n_sub);
    unsigned *p_sub_ind = sub_mesh_index->memptr();

    for (arma::uword i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        arma::uword n_sub_faces = c[i_sub].n_rows;
        arma::uword n_align = (n_sub_faces % vec_size == 0) ? n_sub_faces : (n_sub_faces / vec_size + 1) * vec_size;
        p_sub_ind[i_sub] = (unsigned)n_out;
        n_out += n_align;
    }

    // Assemble output
    meshR->set_size(n_out, 9);
    dtype *p_mesh_out = meshR->memptr();

    const dtype *p_mtl_in = process_mtl_prop ? mtl_prop->memptr() : nullptr;
    dtype *p_mtl_out = nullptr;
    if (process_mtl_prop)
    {
        mtl_propR->set_size(n_out, 5);
        p_mtl_out = mtl_propR->memptr();
    }

    unsigned *p_mesh_index = nullptr;
    if (mesh_index != nullptr)
    {
        mesh_index->zeros(n_out);
        p_mesh_index = mesh_index->memptr();
    }

    for (arma::uword i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        arma::uword n_sub_faces = c[i_sub].n_rows;  // Number of faces in the sub-mesh
        dtype *p_sub_mesh = c[i_sub].memptr();      // Pointer to sub-mesh data
        arma::uword *pi = face_ind[i_sub].memptr(); // Face index list of current sub-mesh

        // Copy sub-mesh data columns by column
        for (arma::uword i_col = 0; i_col < 9; ++i_col)
        {
            arma::uword offset = i_col * n_out + (arma::uword)p_sub_ind[i_sub];
            std::memcpy(&p_mesh_out[offset], &p_sub_mesh[i_col * n_sub_faces], n_sub_faces * sizeof(dtype));
        }

        // Copy material data
        if (process_mtl_prop)
            for (arma::uword i_col = 0; i_col < 5; ++i_col)
            {
                arma::uword offset_out = i_col * n_out + (arma::uword)p_sub_ind[i_sub];
                arma::uword offset_in = i_col * n_mesh;
                for (arma::uword i_sub_face = 0; i_sub_face < n_sub_faces; ++i_sub_face)
                    p_mtl_out[offset_out + i_sub_face] = p_mtl_in[offset_in + pi[i_sub_face]];
            }

        // Write mesh index
        if (p_mesh_index != nullptr)
        {
            arma::uword offset = (arma::uword)p_sub_ind[i_sub];
            for (arma::uword i_sub_face = 0; i_sub_face < n_sub_faces; ++i_sub_face)
                p_mesh_index[offset + i_sub_face] = (unsigned)pi[i_sub_face] + 1;
        }

        // Add padding data
        if (n_sub_faces % vec_size != 0)
        {
            // Calculate bounding box of current sub-mesh
            arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb(&c[i_sub]);
            dtype *p_box = aabb.memptr();

            dtype x = p_box[0] + (dtype)0.5 * (p_box[1] - p_box[0]);
            dtype y = p_box[2] + (dtype)0.5 * (p_box[3] - p_box[2]);
            dtype z = p_box[4] + (dtype)0.5 * (p_box[5] - p_box[4]);

            arma::uword i_start = (arma::uword)p_sub_ind[i_sub] + n_sub_faces;
            arma::uword i_end = (i_sub == n_sub - 1) ? n_out : (arma::uword)p_sub_ind[i_sub + 1];

            for (arma::uword i_col = 0; i_col < 9; ++i_col)
                for (arma::uword i_pad = i_start; i_pad < i_end; ++i_pad)
                {
                    arma::uword offset = i_col * n_out + i_pad;
                    if (i_col % 3 == 0)
                        p_mesh_out[offset] = x;
                    else if (i_col % 3 == 1)
                        p_mesh_out[offset] = y;
                    else
                        p_mesh_out[offset] = z;

                    if (process_mtl_prop && i_col == 0)
                        p_mtl_out[offset] = (dtype)1.0;
                    else if (process_mtl_prop && i_col < 5)
                        p_mtl_out[offset] = (dtype)0.0;
                }
        }
    }

    return n_sub;
}

template arma::uword quadriga_lib::triangle_mesh_segmentation(const arma::Mat<float> *mesh, arma::Mat<float> *meshR,
                                                              arma::u32_vec *sub_mesh_index, arma::uword target_size, arma::uword vec_size,
                                                              const arma::Mat<float> *mtl_prop, arma::Mat<float> *mtl_propR, arma::u32_vec *mesh_index);

template arma::uword quadriga_lib::triangle_mesh_segmentation(const arma::Mat<double> *mesh, arma::Mat<double> *meshR,
                                                              arma::u32_vec *sub_mesh_index, arma::uword target_size, arma::uword vec_size,
                                                              const arma::Mat<double> *mtl_prop, arma::Mat<double> *mtl_propR, arma::u32_vec *mesh_index);
