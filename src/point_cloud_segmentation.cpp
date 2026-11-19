// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "quadriga_lib_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes at the midpoint.
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are placed at the sub-cloud AABB center.
- Produces a reorganized point array and index maps to track reordering.

## Declaration:
```
arma::uword quadriga_lib::point_cloud_segmentation(
    const arma::Mat<dtype> &points,
    arma::Mat<dtype> &points_out,
    arma::u32_vec &sub_cloud_ind,
    arma::uword target_size = 1024,
    arma::uword vec_size = 1,
    arma::u32_vec *forward_ind = nullptr,
    arma::u32_vec *reverse_ind = nullptr);
```

## Inputs:
- **`points`** — Original 3D point cloud; `[n_points, 3]`
- **`target_size`** *(optional)* — Maximum points per sub-cloud before padding
- **`vec_size`** *(optional)* — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of this value; no padding when `1`;
  [[ray_point_intersect]] accepts any value, padding (e.g. 8 for AVX2) only improves its speed

## Outputs:
- **`points_out`** — Reorganized point cloud with points grouped by sub-cloud; `[n_points_out, 3]`
- **`sub_cloud_ind`** — 0-based starting index of each sub-cloud within `points_out`; `[n_sub]`
- **`forward_ind`** *(optional)* — 1-based row in `points` of each row of `points_out`; padding entries are `0`; `[n_points_out]`
- **`reverse_ind`** *(optional)* — 0-based row in `points_out` of each row of `points`; `[n_points]`

## Returns:
- Number of generated sub-clouds `n_sub`

## See also:
- [[point_cloud_aabb]] (bounding box computation)
- [[point_cloud_split]] (related spatial splitting)
- [[ray_point_intersect]] (downstream use case)
MD!*/

// Reorganize a point cloud into smaller sub-clouds for faster processing
template <typename dtype>
arma::uword quadriga_lib::point_cloud_segmentation(const arma::Mat<dtype> &points,
                                                   arma::Mat<dtype> &points_out,
                                                   arma::u32_vec &sub_cloud_ind,
                                                   arma::uword target_size,
                                                   arma::uword vec_size,
                                                   arma::u32_vec *forward_ind,
                                                   arma::u32_vec *reverse_ind)
{

    // Input validation
    if (points.n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns containing x,y,z coordinates.");
    if (points.n_rows == 0)
        throw std::invalid_argument("Input 'points' must have at least one face.");

    arma::uword n_points = points.n_rows;

    qd_check_alias(&points_out, "points_out", &points);
    qd_check_alias_list({&sub_cloud_ind, forward_ind, reverse_ind}, {"sub_cloud_ind", "forward_ind", "reverse_ind"});

    if (target_size == 0)
        throw std::invalid_argument("Input 'target_size' cannot be 0.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    // Create a vector of sub-clouds
    std::vector<arma::Mat<dtype>> c; // Vector of sub-clouds

    // Add first item (creates a copy of the data)
    c.push_back(points);

    // Create base index (0-based)
    std::vector<arma::uvec> fwd_ind; // Index list
    {
        arma::uvec base_index(points.n_rows, arma::fill::none);
        arma::uword *p = base_index.memptr();
        for (arma::uword i = 0; i < n_points; ++i)
            p[i] = i;
        fwd_ind.push_back(base_index);
    }

    // Iterate through all elements
    for (auto sub_cloud_it = c.begin(); sub_cloud_it != c.end();)
    {
        arma::uword n_sub_points = (arma::uword)(*sub_cloud_it).n_rows;
        if (n_sub_points > target_size)
        {
            arma::Mat<dtype> pointsA, pointsB;
            arma::s32_vec split_ind;

            // Split on longest axis
            int split_success = quadriga_lib::point_cloud_split(*sub_cloud_it, pointsA, pointsB, 0, &split_ind);

            // Check the split proportions, must have at least a 10 / 90 split
            float p = (split_success > 0) ? (float)pointsA.n_rows / (float)n_sub_points : 0.5f;
            split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

            // Attempt to split along another axis if failed for longest axis
            int first_test = std::abs(split_success);
            if (split_success < 0) // Failed condition
            {
                // Test second axis
                if (first_test == 2 || first_test == 3) // Test x-axis
                    split_success = quadriga_lib::point_cloud_split(*sub_cloud_it, pointsA, pointsB, 1, &split_ind);
                else // Test y-axis
                    split_success = quadriga_lib::point_cloud_split(*sub_cloud_it, pointsA, pointsB, 2, &split_ind);

                p = (split_success > 0) ? (float)pointsA.n_rows / (float)n_sub_points : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

                // If we still failed, we test the third axis
                if ((first_test == 1 && split_success == -2) || (first_test == 2 && split_success == -1))
                    split_success = quadriga_lib::point_cloud_split(*sub_cloud_it, pointsA, pointsB, 3, &split_ind);
                else if (first_test == 3 && split_success == -1)
                    split_success = quadriga_lib::point_cloud_split(*sub_cloud_it, pointsA, pointsB, 2, &split_ind);

                p = (split_success > 0) ? (float)pointsA.n_rows / (float)n_sub_points : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;
            }

            // Update the point cloud data in memory
            int *ps = split_ind.memptr();
            if (split_success > 0)
            {
                // Split the index list
                arma::uword i_sub = sub_cloud_it - c.begin(); // Sub-cloud index
                auto fwd_ind_it = fwd_ind.begin() + i_sub;    // Forward index iterator
                arma::uword *pi = (*fwd_ind_it).memptr();     // Current cloud index list

                arma::uvec pointsA_index(pointsA.n_rows, arma::fill::none);
                arma::uvec pointsB_index(pointsB.n_rows, arma::fill::none);
                arma::uword *pA = pointsA_index.memptr(); // New face index list of mesh A
                arma::uword *pB = pointsB_index.memptr(); // New face index list of mesh B

                arma::uword iA = 0, iB = 0;
                for (arma::uword i_point = 0; i_point < n_sub_points; ++i_point)
                {
                    if (ps[i_point] == 1)
                        pA[iA++] = pi[i_point];
                    else if (ps[i_point] == 2)
                        pB[iB++] = pi[i_point];
                }

                fwd_ind.erase(fwd_ind_it);
                fwd_ind.push_back(std::move(pointsA_index));
                fwd_ind.push_back(std::move(pointsB_index));

                // Update the vector of sub-meshes
                c.erase(sub_cloud_it);
                c.push_back(std::move(pointsA));
                c.push_back(std::move(pointsB));
                sub_cloud_it = c.begin();
            }
            else
                ++sub_cloud_it;
        }
        else
            ++sub_cloud_it;
    }

    // Get the sub-cloud indices
    arma::uword n_sub = c.size(), n_out = 0;
    sub_cloud_ind.set_size(n_sub);
    unsigned *p_sub_ind = sub_cloud_ind.memptr();

    for (arma::uword i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        arma::uword n_sub_points = c[i_sub].n_rows;
        arma::uword n_align = (n_sub_points % vec_size == 0) ? n_sub_points : (n_sub_points / vec_size + 1) * vec_size;
        p_sub_ind[i_sub] = (unsigned)n_out;
        n_out += n_align;
    }

    // Assemble output
    points_out.set_size(n_out, 3);
    dtype *p_points_out = points_out.memptr();

    unsigned *p_forward_index = nullptr;
    if (forward_ind != nullptr || reverse_ind != nullptr)
        p_forward_index = new unsigned[n_out]();

    for (arma::uword i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        arma::uword n_sub_points = c[i_sub].n_rows; // Number of points in the sub-cloud
        dtype *p_sub_cloud = c[i_sub].memptr();     // Pointer to sub-cloud data
        arma::uword *pi = fwd_ind[i_sub].memptr();  // Point index list of current sub-cloud

        // Copy sub-cloud data columns by column
        for (arma::uword i_col = 0; i_col < 3; ++i_col)
        {
            arma::uword offset = i_col * n_out + (arma::uword)p_sub_ind[i_sub];
            std::memcpy(&p_points_out[offset], &p_sub_cloud[i_col * n_sub_points], n_sub_points * sizeof(dtype));
        }

        // Write index
        if (p_forward_index != nullptr)
        {
            arma::uword offset = (arma::uword)p_sub_ind[i_sub];
            for (arma::uword i_sub_point = 0; i_sub_point < n_sub_points; ++i_sub_point)
                p_forward_index[offset + i_sub_point] = (unsigned)pi[i_sub_point] + 1;
        }

        // Add padding data
        if (n_sub_points % vec_size != 0)
        {
            // Calculate bounding box of current sub-mesh
            arma::Mat<dtype> aabb = quadriga_lib::point_cloud_aabb(c[i_sub]);
            dtype *p_box = aabb.memptr();

            dtype x = p_box[0] + (dtype)0.5 * (p_box[1] - p_box[0]);
            dtype y = p_box[2] + (dtype)0.5 * (p_box[3] - p_box[2]);
            dtype z = p_box[4] + (dtype)0.5 * (p_box[5] - p_box[4]);

            arma::uword i_start = (arma::uword)p_sub_ind[i_sub] + n_sub_points;
            arma::uword i_end = (i_sub == n_sub - 1) ? n_out : (arma::uword)p_sub_ind[i_sub + 1];

            for (arma::uword i_col = 0; i_col < 3; ++i_col)
                for (arma::uword i_pad = i_start; i_pad < i_end; ++i_pad)
                {
                    arma::uword offset = i_col * n_out + i_pad;
                    if (i_col == 0)
                        p_points_out[offset] = x;
                    else if (i_col == 1)
                        p_points_out[offset] = y;
                    else
                        p_points_out[offset] = z;
                }
        }
    }

    // Copy forward index
    if (forward_ind != nullptr)
    {
        forward_ind->set_size(n_out);
        std::memcpy(forward_ind->memptr(), p_forward_index, n_out * sizeof(unsigned));
    }

    // Generate reverse index
    if (reverse_ind != nullptr)
    {
        reverse_ind->set_size(n_points);
        unsigned *p = reverse_ind->memptr();
        for (unsigned i = 0; i < n_out; ++i)
            if (p_forward_index[i] != 0)
                p[p_forward_index[i] - 1] = i;
    }

    if (forward_ind != nullptr || reverse_ind != nullptr)
        delete[] p_forward_index;

    return n_sub;
}

template arma::uword quadriga_lib::point_cloud_segmentation(const arma::Mat<float> &points, arma::Mat<float> &points_out, arma::u32_vec &sub_cloud_ind,
                                                            arma::uword target_size, arma::uword vec_size, arma::u32_vec *forward_ind, arma::u32_vec *reverse_ind);

template arma::uword quadriga_lib::point_cloud_segmentation(const arma::Mat<double> &points, arma::Mat<double> &points_out, arma::u32_vec &sub_cloud_ind,
                                                            arma::uword target_size, arma::uword vec_size, arma::u32_vec *forward_ind, arma::u32_vec *reverse_ind);
