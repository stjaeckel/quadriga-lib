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
# point_cloud_split
Split a point cloud into two sub-clouds along a spatial axis

## Description:
- Splits at the bounding box midpoint along the chosen axis (not the statistical median); 
  the split may be unbalanced if points are non-uniformly distributed.
- If `axis == 0`, the longest bounding box extent is used.
- Returns a negative axis value if the split failed (all points on one side); outputs are not modified in that case.
- Allowed datatypes: `float` or `double`

## Declaration:
```
int quadriga_lib::point_cloud_split(
    const arma::Mat<dtype> *points,
    arma::Mat<dtype> *pointsA,
    arma::Mat<dtype> *pointsB,
    int axis = 0,
    arma::Col<int> *split_ind = nullptr);
```

## Input Arguments:
- **`points`** — Input point cloud; `[n_points, 3]`
- **`axis`** *(optional)* — Split axis: `0` = longest extent, `1` = x, `2` = y, `3` = z

## Output Arguments:
- **`pointsA`** — First sub-cloud; `[n_pointsA, 3]`
- **`pointsB`** — Second sub-cloud; `[n_pointsB, 3]`
- **`split_ind`** *(optional)* — Per-point destination: `1` = pointsA, `2` = pointsB, `0` = error; `[n_points]`

## Returns:
- Axis used: `1` = x, `2` = y, `3` = z; negative (`-1`, `-2`, `-3`) if split failed

## See also:
- [[point_cloud_aabb]] (bounding box computation)
- [[point_cloud_segmentation]] (recursive partitioning using this function)
- [[ray_point_intersect]] (downstream use case)
MD!*/

// Split a point cloud into two sub-clouds along a given axis
template <typename dtype>
int quadriga_lib::point_cloud_split(const arma::Mat<dtype> *points,
                                    arma::Mat<dtype> *pointsA,
                                    arma::Mat<dtype> *pointsB,
                                    int axis,
                                    arma::Col<int> *split_ind)
{
    if (points == nullptr)
        throw std::invalid_argument("Input 'points' cannot be NULL.");
    if (pointsA == nullptr)
        throw std::invalid_argument("Output 'pointsA' cannot be NULL.");
    if (pointsB == nullptr)
        throw std::invalid_argument("Output 'pointsB' cannot be NULL.");

    // Calculate bounding box
    arma::Mat<dtype> aabb = quadriga_lib::point_cloud_aabb(points);

    size_t n_points_t = (size_t)points->n_rows;
    size_t n_values_t = (size_t)points->n_elem;
    const dtype *p_points = points->memptr();

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

    // Determine all points that are outside box A
    bool *isB = new bool[n_points_t](); // Init to false

    for (size_t i_val = 0; i_val < n_values_t; ++i_val)
    {
        dtype v = p_points[i_val];         // Mesh value
        size_t i_col = i_val / n_points_t; // Column index in mesh
        size_t i_row = i_val % n_points_t; // Row index in mesh

        if (i_col == 0)
            isB[i_row] = (v > x_max) ? true : isB[i_row];
        else if (i_col == 1)
            isB[i_row] = (v > y_max) ? true : isB[i_row];
        else
            isB[i_row] = (v > z_max) ? true : isB[i_row];
    }

    // Count items in both sub-meshes
    size_t n_pointsA = 0, n_pointsB = 0;
    for (size_t i = 0; i < n_points_t; ++i)
        if (isB[i])
            ++n_pointsB;
        else
            ++n_pointsA;

    // Check if the mesh was split
    if (n_pointsA == 0 || n_pointsB == 0)
        return -axis;

    // Adjust output size
    pointsA->set_size(n_pointsA, 3);
    pointsB->set_size(n_pointsB, 3);

    dtype *p_pointsA = pointsA->memptr();
    dtype *p_pointsB = pointsB->memptr();

    bool write_split_ind = false;
    int *p_split_ind = nullptr;
    if (split_ind != nullptr)
    {
        write_split_ind = true;
        if (split_ind->n_elem != points->n_rows)
            split_ind->zeros(points->n_rows);
        else
            split_ind->zeros();
        p_split_ind = split_ind->memptr();
    }

    // Copy data
    size_t i_pointA = 0, i_pointB = 0;
    for (size_t i_val = 0; i_val < n_values_t; ++i_val)
    {
        dtype v = p_points[i_val];         // Mesh value
        size_t i_row = i_val % n_points_t; // Row index in mesh

        if (isB[i_row])
        {
            p_pointsB[i_pointB++] = v;
            if (write_split_ind)
                p_split_ind[i_row] = 2;
        }
        else
        {
            p_pointsA[i_pointA++] = v;
            if (write_split_ind)
                p_split_ind[i_row] = 1;
        }
    }

    delete[] isB;
    return axis;
}

template int quadriga_lib::point_cloud_split(const arma::Mat<float> *points, arma::Mat<float> *pointsA, arma::Mat<float> *pointsB, int axis, arma::Col<int> *split_ind);

template int quadriga_lib::point_cloud_split(const arma::Mat<double> *points, arma::Mat<double> *pointsA, arma::Mat<double> *pointsB, int axis, arma::Col<int> *split_ind);
