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
# point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

## Description:
- Each row of the output contains `[x_min, x_max, y_min, y_max, z_min, z_max]` for one sub-cloud
- If `sub_cloud_index` is `nullptr` or empty, the entire input is treated as a single cloud; last index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros
- Allowed datatypes: `float` or `double`

## Declaration:
```
arma::Mat<dtype> quadriga_lib::point_cloud_aabb(
    const arma::Mat<dtype> *points,
    const arma::u32_vec *sub_cloud_index = nullptr,
    arma::uword vec_size = 1);
```

## Input Arguments:
- **`points`** — 3D point coordinates; `[n_points, 3]`
- **`sub_cloud_index`** *(optional)* — Row indices marking the start of each sub-cloud; use [[point_cloud_segmentation]] to generate; `[n_sub]`
- **`vec_size`** *(optional)* — SIMD alignment padding factor (e.g. 4, 8, 16)

## Returns:
- Bounding box matrix; `[n_out, 6]` where `n_out` is `n_sub` padded to a multiple of `vec_size`

## See also:
- [[point_cloud_segmentation]] (generate sub-cloud indices)
- [[point_cloud_split]] (split point cloud)
- [[ray_point_intersect]] (use AABBs for intersection)
MD!*/

template <typename dtype>
arma::Mat<dtype> quadriga_lib::point_cloud_aabb(const arma::Mat<dtype> *points,
                                                const arma::u32_vec *sub_cloud_index,
                                                arma::uword vec_size)
{
    // Input validation
    if (points == nullptr)
        throw std::invalid_argument("Input 'points' cannot be NULL.");
    if (points->n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns containing x,y,z coordinates.");
    if (points->n_rows == 0)
        throw std::invalid_argument("Input 'points' must have at least one entry.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    arma::uword n_points = (arma::uword)points->n_rows;
    arma::uword n_values = (arma::uword)points->n_elem;
    const dtype *p_points = points->memptr();

    const unsigned first_sub_cloud_ind = 0;
    const unsigned *p_sub = &first_sub_cloud_ind;
    arma::uword n_sub = 1;

    if (sub_cloud_index != nullptr && sub_cloud_index->n_elem > 0)
    {
        n_sub = (arma::uword)sub_cloud_index->n_elem;
        p_sub = sub_cloud_index->memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-cloud must start at index 0.");

        for (arma::uword i = 1; i < n_sub; ++i)
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-cloud indices must be sorted in ascending order.");

        if (p_sub[n_sub - 1] >= (unsigned)n_points)
            throw std::invalid_argument("Sub-cloud indices cannot exceed number of points.");
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

    arma::uword i_sub = 0, i_next = n_points;
    for (arma::uword i_point = 0; i_point < n_values; ++i_point)
    {
        dtype v = p_points[i_point];            // Point value
        arma::uword i_col = i_point / n_points; // Column index in mesh
        arma::uword i_row = i_point % n_points; // Row index in mesh

        if (i_row == 0)
        {
            i_sub = 0;
            i_next = (i_sub == n_sub - 1) ? n_points : (arma::uword)p_sub[i_sub + 1];
        }
        else if (i_row == i_next)
        {
            ++i_sub;
            i_next = (i_sub == n_sub - 1) ? n_points : (arma::uword)p_sub[i_sub + 1];
        }

        if (i_col == 0)
        {
            x_min[i_sub] = (v < x_min[i_sub]) ? v : x_min[i_sub];
            x_max[i_sub] = (v > x_max[i_sub]) ? v : x_max[i_sub];
        }
        else if (i_col == 1)
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

template arma::Mat<float> quadriga_lib::point_cloud_aabb(const arma::Mat<float> *points,
                                                         const arma::u32_vec *sub_cloud_index,
                                                         arma::uword vec_size);

template arma::Mat<double> quadriga_lib::point_cloud_aabb(const arma::Mat<double> *points,
                                                          const arma::u32_vec *sub_cloud_index,
                                                          arma::uword vec_size);
