// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"
#include "qd_helpers_site.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// The point cloud functions (n_col = 3, one point per row) and the triangle mesh functions (n_col = 9, three vertices
// per row) share their code; column c of a row holds coordinate c % 3.

// Checks 'points' [n_points, 3] (n_col = 3) or 'mesh' [n_mesh, 9] (n_col = 9; further columns allowed unless 'exact')
template <arma::uword n_col, typename dtype>
static void qd_check_partition_input(const arma::Mat<dtype> &data, bool exact)
{
    if (n_col == 3 && data.n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns containing x,y,z coordinates.");
    if (n_col == 9 && exact && data.n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (n_col == 9 && data.n_cols < 9)
        throw std::invalid_argument("Input 'mesh' must have at least 9 columns containing x,y,z coordinates of 3 vertices.");
    if (data.n_rows == 0)
        throw std::invalid_argument(n_col == 3 ? "Input 'points' cannot be empty." : "Input 'mesh' cannot be empty.");
}

// Bounding boxes [n_out, 6] of the sub-objects that start at the rows 'p_sub' [n_sub] of 'data' (first n_col columns);
// rows n_sub to n_out - 1 (padding to a multiple of 'vec_size') are zero
template <arma::uword n_col, typename dtype>
static arma::Mat<dtype> qd_partition_aabb(const arma::Mat<dtype> &data, const unsigned *p_sub, arma::uword n_sub, arma::uword vec_size)
{
    const arma::uword n_rows = data.n_rows;
    const arma::uword n_values = n_col * n_rows; // Column-major, further columns are ignored
    const dtype *p_data = data.memptr();

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

    arma::uword i_sub = 0, i_next = n_rows;
    for (arma::uword i_val = 0; i_val < n_values; ++i_val)
    {
        dtype v = p_data[i_val];
        arma::uword i_col = i_val / n_rows;
        arma::uword i_row = i_val % n_rows;

        if (i_row == 0)
        {
            i_sub = 0;
            i_next = (i_sub == n_sub - 1) ? n_rows : (arma::uword)p_sub[i_sub + 1];
        }
        else if (i_row == i_next)
        {
            ++i_sub;
            i_next = (i_sub == n_sub - 1) ? n_rows : (arma::uword)p_sub[i_sub + 1];
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

// Splits 'data' (first n_col columns) at the center of its bounding box along 'axis' (0 = longest extent, 1 = x,
// 2 = y, 3 = z) into A and B; rows with a vertex above the center go to B. Returns the axis, or its negative when one
// part would be empty (outputs then untouched). 'split_ind' [n_rows] (optional): 1 = A, 2 = B.
template <arma::uword n_col, typename dtype>
static int qd_partition_split(const arma::Mat<dtype> &data, arma::Mat<dtype> &A, arma::Mat<dtype> &B, int axis, arma::Col<int> *split_ind)
{
    const unsigned first_row = 0;
    const arma::Mat<dtype> aabb = qd_partition_aabb<n_col>(data, &first_row, 1, 1);

    const size_t n_rows = (size_t)data.n_rows;
    const size_t n_values = n_col * n_rows;
    const dtype *p_data = data.memptr();

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

    // Rows outside box A
    std::vector<char> isB(n_rows, 0);
    for (size_t i_val = 0; i_val < n_values; ++i_val)
    {
        dtype v = p_data[i_val];
        size_t i_col = i_val / n_rows;
        size_t i_row = i_val % n_rows;

        if (i_col % 3 == 0)
            isB[i_row] = (v > x_max) ? 1 : isB[i_row];
        else if (i_col % 3 == 1)
            isB[i_row] = (v > y_max) ? 1 : isB[i_row];
        else
            isB[i_row] = (v > z_max) ? 1 : isB[i_row];
    }

    size_t n_B = 0;
    for (size_t i = 0; i < n_rows; ++i)
        n_B += isB[i] ? 1 : 0;
    size_t n_A = n_rows - n_B;

    if (n_A == 0 || n_B == 0)
        return -axis;

    A.set_size(n_A, n_col);
    B.set_size(n_B, n_col);
    dtype *p_A = A.memptr(), *p_B = B.memptr();

    int *p_split_ind = nullptr;
    if (split_ind != nullptr)
    {
        split_ind->set_size(data.n_rows);
        p_split_ind = split_ind->memptr();
    }

    size_t i_A = 0, i_B = 0;
    for (size_t i_val = 0; i_val < n_values; ++i_val)
    {
        dtype v = p_data[i_val];
        size_t i_row = i_val % n_rows;

        if (isB[i_row])
            p_B[i_B++] = v;
        else
            p_A[i_A++] = v;

        if (p_split_ind)
            p_split_ind[i_row] = isB[i_row] ? 2 : 1;
    }

    return axis;
}

// Splits 'data' [n_rows, n_col] recursively until each sub-object has at most 'target_size' rows (a split must leave
// at least 10 % on each side) and writes the sub-objects to 'data_out' [n_out, n_col], each padded to a multiple of
// 'vec_size' with rows at the center of its bounding box. 'sub_ind' [n_sub] gets the 0-based start rows, 'fwd_ind'
// [n_out] (optional) the 1-based row in 'data' of each output row (0 = padding). Returns n_sub.
template <arma::uword n_col, typename dtype>
static arma::uword qd_partition_segmentation(const arma::Mat<dtype> &data, arma::Mat<dtype> &data_out, arma::u32_vec &sub_ind,
                                             arma::uword target_size, arma::uword vec_size, arma::u32_vec *fwd_ind)
{
    std::vector<arma::Mat<dtype>> c; // Sub-objects
    c.push_back(data);

    std::vector<arma::uvec> row_ind; // 0-based rows in 'data' of each sub-object
    row_ind.push_back(arma::regspace<arma::uvec>(0, data.n_rows - 1));

    for (auto sub_it = c.begin(); sub_it != c.end();)
    {
        arma::uword n_sub_rows = sub_it->n_rows;
        if (n_sub_rows <= target_size)
        {
            ++sub_it;
            continue;
        }

        arma::Mat<dtype> A, B;
        arma::s32_vec split_ind;

        // Split along 'ax'; an unbalanced split (less than 10 / 90) counts as failed
        auto try_split = [&](int ax)
        {
            int s = qd_partition_split<n_col>(*sub_it, A, B, ax, &split_ind);
            float p = (s > 0) ? (float)A.n_rows / (float)n_sub_rows : 0.5f;
            return (p < 0.1f || p > 0.9f) ? -s : s;
        };

        // Longest axis first, then the other two
        int split_success = try_split(0);
        int first_test = std::abs(split_success);
        if (split_success < 0)
        {
            split_success = try_split((first_test == 2 || first_test == 3) ? 1 : 2);
            if ((first_test == 1 && split_success == -2) || (first_test == 2 && split_success == -1))
                split_success = try_split(3);
            else if (first_test == 3 && split_success == -1)
                split_success = try_split(2);
        }

        if (split_success <= 0)
        {
            ++sub_it;
            continue;
        }

        const int *ps = split_ind.memptr();
        arma::uword i_sub = sub_it - c.begin();
        auto row_it = row_ind.begin() + i_sub;
        const arma::uword *pi = row_it->memptr();

        arma::uvec A_index(A.n_rows, arma::fill::none), B_index(B.n_rows, arma::fill::none);
        arma::uword *pA = A_index.memptr(), *pB = B_index.memptr();
        arma::uword iA = 0, iB = 0;
        for (arma::uword i_row = 0; i_row < n_sub_rows; ++i_row)
        {
            if (ps[i_row] == 1)
                pA[iA++] = pi[i_row];
            else if (ps[i_row] == 2)
                pB[iB++] = pi[i_row];
        }

        row_ind.erase(row_it);
        row_ind.push_back(std::move(A_index));
        row_ind.push_back(std::move(B_index));

        c.erase(sub_it);
        c.push_back(std::move(A));
        c.push_back(std::move(B));
        sub_it = c.begin();
    }

    arma::uword n_sub = c.size(), n_out = 0;
    sub_ind.set_size(n_sub);
    unsigned *p_sub_ind = sub_ind.memptr();

    for (arma::uword i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        arma::uword n_sub_rows = c[i_sub].n_rows;
        arma::uword n_align = (n_sub_rows % vec_size == 0) ? n_sub_rows : (n_sub_rows / vec_size + 1) * vec_size;
        p_sub_ind[i_sub] = (unsigned)n_out;
        n_out += n_align;
    }

    data_out.set_size(n_out, n_col);
    dtype *p_out = data_out.memptr();

    unsigned *p_fwd = nullptr;
    if (fwd_ind != nullptr)
    {
        fwd_ind->zeros(n_out);
        p_fwd = fwd_ind->memptr();
    }

    for (arma::uword i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        arma::uword n_sub_rows = c[i_sub].n_rows;
        const dtype *p_sub_data = c[i_sub].memptr();
        const arma::uword *pi = row_ind[i_sub].memptr();
        arma::uword i_start = (arma::uword)p_sub_ind[i_sub];

        for (arma::uword i_col = 0; i_col < n_col; ++i_col)
            std::memcpy(&p_out[i_col * n_out + i_start], &p_sub_data[i_col * n_sub_rows], n_sub_rows * sizeof(dtype));

        if (p_fwd != nullptr)
            for (arma::uword i_row = 0; i_row < n_sub_rows; ++i_row)
                p_fwd[i_start + i_row] = (unsigned)pi[i_row] + 1;

        // Padding rows (for meshes: zero-sized triangles) at the bounding box center of the sub-object
        if (n_sub_rows % vec_size != 0)
        {
            const unsigned first_row = 0;
            arma::Mat<dtype> aabb = qd_partition_aabb<n_col>(c[i_sub], &first_row, 1, 1);
            const dtype *p_box = aabb.memptr();
            const dtype center[3] = {p_box[0] + (dtype)0.5 * (p_box[1] - p_box[0]),
                                     p_box[2] + (dtype)0.5 * (p_box[3] - p_box[2]),
                                     p_box[4] + (dtype)0.5 * (p_box[5] - p_box[4])};

            arma::uword i_end = (i_sub == n_sub - 1) ? n_out : (arma::uword)p_sub_ind[i_sub + 1];
            for (arma::uword i_col = 0; i_col < n_col; ++i_col)
                for (arma::uword i_pad = i_start + n_sub_rows; i_pad < i_end; ++i_pad)
                    p_out[i_col * n_out + i_pad] = center[i_col % 3];
        }
    }

    return n_sub;
}

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

- Each row of the output contains `[x_min, x_max, y_min, y_max, z_min, z_max]` for one sub-cloud
- If `sub_cloud_ind` is empty, the entire input is treated as a single cloud; last index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

## Declaration:
```
arma::Mat<dtype> quadriga_lib::point_cloud_aabb(
    const arma::Mat<dtype> &points,
    const arma::u32_vec &sub_cloud_ind = arma::u32_vec(),
    arma::uword vec_size = 1);
```

## Inputs:
- **`points`** — 3D point coordinates; `[n_points, 3]`
- **`sub_cloud_ind`** *(optional)* — Row indices marking the start of each sub-cloud; use [[point_cloud_segmentation]] to generate; `[n_sub]`
- **`vec_size`** *(optional)* — SIMD alignment padding factor (e.g. 4, 8, 16)

## Returns:
- Bounding boxes `aabb`, one row per sub-cloud; `[n_out, 6]` where `n_out` is `n_sub` padded to a multiple of `vec_size`

## See also:
- [[point_cloud_segmentation]] (generate sub-cloud indices)
- [[point_cloud_split]] (split point cloud)
- [[ray_point_intersect]] (use AABBs for intersection)
MD!*/

template <typename dtype>
arma::Mat<dtype> quadriga_lib::point_cloud_aabb(const arma::Mat<dtype> &points,
                                                const arma::u32_vec &sub_cloud_ind,
                                                arma::uword vec_size)
{
    qd_check_partition_input<3>(points, true);
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    const unsigned first_row = 0;
    if (qd_check_sub_ind(sub_cloud_ind, points.n_rows, "sub_cloud_ind", "point") == 0)
        return qd_partition_aabb<3>(points, &first_row, 1, vec_size);
    return qd_partition_aabb<3>(points, sub_cloud_ind.memptr(), sub_cloud_ind.n_elem, vec_size);
}

template arma::Mat<float> quadriga_lib::point_cloud_aabb(const arma::Mat<float> &points,
                                                         const arma::u32_vec &sub_cloud_ind,
                                                         arma::uword vec_size);

template arma::Mat<double> quadriga_lib::point_cloud_aabb(const arma::Mat<double> &points,
                                                          const arma::u32_vec &sub_cloud_ind,
                                                          arma::uword vec_size);

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
    qd_check_partition_input<9>(mesh, false);
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    const unsigned first_row = 0;
    if (qd_check_sub_ind(sub_mesh_ind, mesh.n_rows, "sub_mesh_ind", "mesh face") == 0)
        return qd_partition_aabb<9>(mesh, &first_row, 1, vec_size);
    return qd_partition_aabb<9>(mesh, sub_mesh_ind.memptr(), sub_mesh_ind.n_elem, vec_size);
}

template arma::Mat<float> quadriga_lib::triangle_mesh_aabb(const arma::Mat<float> &mesh,
                                                           const arma::u32_vec &sub_mesh_ind,
                                                           arma::uword vec_size);

template arma::Mat<double> quadriga_lib::triangle_mesh_aabb(const arma::Mat<double> &mesh,
                                                            const arma::u32_vec &sub_mesh_ind,
                                                            arma::uword vec_size);

/*!MD
# point_cloud_split
Split a point cloud into two sub-clouds along a spatial axis

- Splits at the bounding box midpoint along the chosen axis (not the statistical median); 
  the split may be unbalanced if points are non-uniformly distributed.
- If `axis == 0`, the longest bounding box extent is used.
- Returns a negative axis value if the split failed (all points on one side); outputs are not modified in that case.

## Declaration:
```
int quadriga_lib::point_cloud_split(
    const arma::Mat<dtype> &points,
    arma::Mat<dtype> &pointsA,
    arma::Mat<dtype> &pointsB,
    int axis = 0,
    arma::Col<int> *split_ind = nullptr);
```

## Inputs:
- **`points`** — Input point cloud; `[n_points, 3]`
- **`axis`** *(optional)* — Split axis: `0` = longest extent, `1` = x, `2` = y, `3` = z

## Outputs:
- **`pointsA`** — First sub-cloud; `[n_pointsA, 3]`
- **`pointsB`** — Second sub-cloud; `[n_pointsB, 3]`
- **`split_ind`** *(optional)* — Per-point destination: `1` = pointsA, `2` = pointsB; `[n_points]`

## Returns:
- Axis used: `1` = x, `2` = y, `3` = z; negative (`-1`, `-2`, `-3`) if split failed

## See also:
- [[point_cloud_aabb]] (bounding box computation)
- [[point_cloud_segmentation]] (recursive partitioning using this function)
- [[ray_point_intersect]] (downstream use case)
MD!*/

template <typename dtype>
int quadriga_lib::point_cloud_split(const arma::Mat<dtype> &points,
                                    arma::Mat<dtype> &pointsA,
                                    arma::Mat<dtype> &pointsB,
                                    int axis,
                                    arma::Col<int> *split_ind)
{
    qd_check_alias_list({&pointsA, &pointsB}, {"pointsA", "pointsB"}, &points);
    qd_check_partition_input<3>(points, true);
    return qd_partition_split<3>(points, pointsA, pointsB, axis, split_ind);
}

template int quadriga_lib::point_cloud_split(const arma::Mat<float> &points, arma::Mat<float> &pointsA, arma::Mat<float> &pointsB, int axis, arma::Col<int> *split_ind);

template int quadriga_lib::point_cloud_split(const arma::Mat<double> &points, arma::Mat<double> &pointsA, arma::Mat<double> &pointsB, int axis, arma::Col<int> *split_ind);

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
    const arma::Mat<dtype> &mesh,
    arma::Mat<dtype> &meshA,
    arma::Mat<dtype> &meshB,
    int axis = 0,
    arma::Col<int> *split_ind = nullptr);
```

## Inputs:
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; further columns are ignored and not copied to the outputs; `[n_mesh, 9]`
- **`axis`** *(optional)* — Split axis: 0 = longest extent, 1 = x, 2 = y, 3 = z

## Outputs:
- **`meshA`** — Triangles with all vertices within the lower half of the bounding box; `[n_meshA, 9]`
- **`meshB`** — Triangles with at least one vertex exceeding the split threshold; `[n_meshB, 9]`
- **`split_ind`** *(optional)* — Per-triangle assignment: 1 = meshA, 2 = meshB; `[n_mesh]`

## Returns:
- Axis used for the split (1, 2, or 3); negative (-1, -2, or -3) on failure

## See also:
- [[triangle_mesh_segmentation]] (calls this function recursively)
MD!*/

template <typename dtype>
int quadriga_lib::triangle_mesh_split(const arma::Mat<dtype> &mesh,
                                      arma::Mat<dtype> &meshA,
                                      arma::Mat<dtype> &meshB,
                                      int axis,
                                      arma::Col<int> *split_ind)
{
    qd_check_alias_list({&meshA, &meshB}, {"meshA", "meshB"}, &mesh);
    qd_check_partition_input<9>(mesh, false);
    return qd_partition_split<9>(mesh, meshA, meshB, axis, split_ind);
}

template int quadriga_lib::triangle_mesh_split(const arma::Mat<float> &mesh,
                                               arma::Mat<float> &meshA,
                                               arma::Mat<float> &meshB,
                                               int axis,
                                               arma::Col<int> *split_ind);

template int quadriga_lib::triangle_mesh_split(const arma::Mat<double> &mesh,
                                               arma::Mat<double> &meshA,
                                               arma::Mat<double> &meshB,
                                               int axis,
                                               arma::Col<int> *split_ind);

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

template <typename dtype>
arma::uword quadriga_lib::point_cloud_segmentation(const arma::Mat<dtype> &points,
                                                   arma::Mat<dtype> &points_out,
                                                   arma::u32_vec &sub_cloud_ind,
                                                   arma::uword target_size,
                                                   arma::uword vec_size,
                                                   arma::u32_vec *forward_ind,
                                                   arma::u32_vec *reverse_ind)
{
    qd_check_partition_input<3>(points, true);
    qd_check_alias(&points_out, "points_out", &points);
    qd_check_alias_list({&sub_cloud_ind, forward_ind, reverse_ind}, {"sub_cloud_ind", "forward_ind", "reverse_ind"});

    if (target_size == 0)
        throw std::invalid_argument("Input 'target_size' cannot be 0.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    arma::u32_vec fwd_local;
    arma::u32_vec *fwd = (forward_ind != nullptr) ? forward_ind : (reverse_ind != nullptr ? &fwd_local : nullptr);
    arma::uword n_sub = qd_partition_segmentation<3>(points, points_out, sub_cloud_ind, target_size, vec_size, fwd);

    if (reverse_ind != nullptr)
    {
        reverse_ind->set_size(points.n_rows);
        unsigned *p = reverse_ind->memptr();
        const unsigned *p_fwd = fwd->memptr();
        for (arma::uword i = 0; i < fwd->n_elem; ++i)
            if (p_fwd[i] != 0)
                p[p_fwd[i] - 1] = (unsigned)i;
    }

    return n_sub;
}

template arma::uword quadriga_lib::point_cloud_segmentation(const arma::Mat<float> &points, arma::Mat<float> &points_out, arma::u32_vec &sub_cloud_ind,
                                                            arma::uword target_size, arma::uword vec_size, arma::u32_vec *forward_ind, arma::u32_vec *reverse_ind);

template arma::uword quadriga_lib::point_cloud_segmentation(const arma::Mat<double> &points, arma::Mat<double> &points_out, arma::u32_vec &sub_cloud_ind,
                                                            arma::uword target_size, arma::uword vec_size, arma::u32_vec *forward_ind, arma::u32_vec *reverse_ind);

/*!MD
# triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; `mesh_ind` uses 0 to mark padding entries
- If `mtl_ind` is provided, material indices are reordered and padded the same way (padding uses index 0)

## Declaration:
```
arma::uword triangle_mesh_segmentation(
    const arma::Mat<dtype> &mesh,
    arma::Mat<dtype> &mesh_out,
    arma::u32_vec &sub_mesh_ind,
    arma::uword target_size = 1024,
    arma::uword vec_size = 1,
    const arma::uvec &mtl_ind = arma::uvec(),
    arma::uvec *mtl_ind_out = nullptr,
    arma::u32_vec *mesh_ind = nullptr);
```

## Inputs:
- **`mesh`** — Triangle vertices, each row `{x1,y1,z1,x2,y2,z2,x3,y3,z3}`; `[n_mesh, 9]`
- **`target_size`** *(optional)* — Target triangle count per sub-mesh; for best performance set near `sqrt(n_mesh)`
- **`vec_size`** *(optional)* — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each sub-mesh row count rounded up to a multiple of this value
- **`mtl_ind`** *(optional)* — Material indices per triangle (the `csv_ind` output of [[obj_file_read]]); `[n_mesh]`

## Outputs:
- **`mesh_out`** — Reordered and padded triangle vertices; `[n_mesh_out, 9]`
- **`sub_mesh_ind`** — 0-based start indices of sub-meshes in `mesh_out`; `[n_sub]`
- **`mtl_ind_out`** *(optional)* — Reordered and padded material indices (padding = 0); `[n_mesh_out]`
- **`mesh_ind`** *(optional)* — 1-based row in `mesh` of each row of `mesh_out` (0 = padding); `[n_mesh_out]`

## Returns:
- Number of created sub-meshes `n_sub`

## See also:
- [[calc_diffraction_gain]] (uses `sub_mesh_ind` for acceleration)
- [[obj_file_read]] (defines `mtl_ind` / `csv_ind`)
MD!*/

template <typename dtype>
arma::uword quadriga_lib::triangle_mesh_segmentation(const arma::Mat<dtype> &mesh,
                                                     arma::Mat<dtype> &mesh_out,
                                                     arma::u32_vec &sub_mesh_ind,
                                                     arma::uword target_size,
                                                     arma::uword vec_size,
                                                     const arma::uvec &mtl_ind,
                                                     arma::uvec *mtl_ind_out,
                                                     arma::u32_vec *mesh_ind)
{
    qd_check_partition_input<9>(mesh, true);
    qd_check_alias(&mesh_out, "mesh_out", &mesh);
    qd_check_alias(mtl_ind_out, "mtl_ind_out", &mtl_ind);
    qd_check_alias_outputs(&sub_mesh_ind, "sub_mesh_ind", mesh_ind, "mesh_ind");

    if (target_size == 0)
        throw std::invalid_argument("Input 'target_size' cannot be 0.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    bool process_mtl_ind = (mtl_ind.n_elem != 0) && (mtl_ind_out != nullptr);

    if (process_mtl_ind && mtl_ind.n_elem != mesh.n_rows)
        throw std::invalid_argument("Number of faces in 'mesh' and length of 'mtl_ind' do not match.");

    arma::u32_vec fwd_local;
    arma::u32_vec *fwd = (mesh_ind != nullptr) ? mesh_ind : (process_mtl_ind ? &fwd_local : nullptr);
    arma::uword n_sub = qd_partition_segmentation<9>(mesh, mesh_out, sub_mesh_ind, target_size, vec_size, fwd);

    // Padding rows get material index 0 (transparent fallback / air)
    if (process_mtl_ind)
    {
        mtl_ind_out->set_size(fwd->n_elem);
        arma::uword *p_mtl_out = mtl_ind_out->memptr();
        const arma::uword *p_mtl_in = mtl_ind.memptr();
        const unsigned *p_fwd = fwd->memptr();
        for (arma::uword i = 0; i < fwd->n_elem; ++i)
            p_mtl_out[i] = p_fwd[i] ? p_mtl_in[p_fwd[i] - 1] : 0;
    }

    return n_sub;
}

template arma::uword quadriga_lib::triangle_mesh_segmentation(const arma::Mat<float> &mesh, arma::Mat<float> &mesh_out,
                                                              arma::u32_vec &sub_mesh_ind, arma::uword target_size, arma::uword vec_size,
                                                              const arma::uvec &mtl_ind, arma::uvec *mtl_ind_out, arma::u32_vec *mesh_ind);

template arma::uword quadriga_lib::triangle_mesh_segmentation(const arma::Mat<double> &mesh, arma::Mat<double> &mesh_out,
                                                              arma::u32_vec &sub_mesh_ind, arma::uword target_size, arma::uword vec_size,
                                                              const arma::uvec &mtl_ind, arma::uvec *mtl_ind_out, arma::u32_vec *mesh_ind);
