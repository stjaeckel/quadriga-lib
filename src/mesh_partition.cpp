// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_tools.hpp"
#include "qd_helpers_shared.hpp"

#include <vector>

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
    // Input validation
    if (points.n_cols != 3)
        throw std::invalid_argument("Input 'points' must have 3 columns containing x,y,z coordinates.");
    if (points.n_rows == 0)
        throw std::invalid_argument("Input 'points' must have at least one entry.");
    if (vec_size == 0)
        throw std::invalid_argument("Input 'vec_size' cannot be 0.");

    arma::uword n_points = (arma::uword)points.n_rows;
    arma::uword n_values = (arma::uword)points.n_elem;
    const dtype *p_points = points.memptr();

    const unsigned first_sub_cloud_ind = 0;
    const unsigned *p_sub = &first_sub_cloud_ind;
    arma::uword n_sub = 1;

    if (sub_cloud_ind.n_elem > 0)
    {
        n_sub = (arma::uword)sub_cloud_ind.n_elem;
        p_sub = sub_cloud_ind.memptr();

        if (*p_sub != 0U)
            throw std::invalid_argument("First sub-cloud must start at the first point.");

        for (arma::uword i = 1; i < n_sub; ++i)
            if (p_sub[i] <= p_sub[i - 1])
                throw std::invalid_argument("Sub-cloud indices must be sorted in ascending order.");

        if (p_sub[n_sub - 1] >= (unsigned)n_points)
            throw std::invalid_argument("Sub-cloud indices cannot exceed number of points.");
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
    return aabb;
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
int quadriga_lib::point_cloud_split(const arma::Mat<dtype> &points,
                                    arma::Mat<dtype> &pointsA,
                                    arma::Mat<dtype> &pointsB,
                                    int axis,
                                    arma::Col<int> *split_ind)
{
    qd_check_alias_list({&pointsA, &pointsB}, {"pointsA", "pointsB"}, &points);

    // Calculate bounding box
    const arma::Mat<dtype> aabb = quadriga_lib::point_cloud_aabb(points);

    size_t n_points_t = (size_t)points.n_rows;
    size_t n_values_t = (size_t)points.n_elem;
    const dtype *p_points = points.memptr();

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
    std::vector<char> isB(n_points_t, 0); // Init to false

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
    pointsA.set_size(n_pointsA, 3);
    pointsB.set_size(n_pointsB, 3);

    dtype *p_pointsA = pointsA.memptr();
    dtype *p_pointsB = pointsB.memptr();

    bool write_split_ind = false;
    int *p_split_ind = nullptr;
    if (split_ind != nullptr)
    {
        write_split_ind = true;
        if (split_ind->n_elem != points.n_rows)
            split_ind->zeros(points.n_rows);
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

    return axis;
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
- **`split_ind`** *(optional)* — Per-triangle assignment: 1 = meshA, 2 = meshB, 0 = unassigned (failure); `[n_mesh]`

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

    // Calculate bounding box
    const arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb(mesh);

    arma::uword n_face = mesh.n_rows;
    arma::uword n_mesh = 9 * n_face; // First 9 columns (column-major), further columns are ignored
    const dtype *p_mesh = mesh.memptr();

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
    meshA.set_size(n_faceA, 9);
    meshB.set_size(n_faceB, 9);

    dtype *p_meshA = meshA.memptr();
    dtype *p_meshB = meshB.memptr();

    bool write_split_ind = false;
    int *p_split_ind = nullptr;
    if (split_ind != nullptr)
    {
        write_split_ind = true;
        if (split_ind->n_elem != mesh.n_rows)
            split_ind->zeros(mesh.n_rows);
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
    // Input validation
    if (mesh.n_cols != 9)
        throw std::invalid_argument("Input 'mesh' must have 9 columns containing x,y,z coordinates of 3 vertices.");
    if (mesh.n_rows == 0)
        throw std::invalid_argument("Input 'mesh' must have at least one face.");

    arma::uword n_mesh = mesh.n_rows;

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

    // Create a vector of meshes
    std::vector<arma::Mat<dtype>> c; // Vector of sub-meshes

    // Add first mesh (creates a copy of the data)
    c.push_back(mesh);

    // Create base index (0-based)
    std::vector<arma::uvec> face_ind; // Index list
    {
        arma::uvec base_index(mesh.n_rows, arma::fill::none);
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
            int split_success = quadriga_lib::triangle_mesh_split(*sub_mesh_it, meshA, meshB, 0, &split_ind);

            // Check the split proportions, must have at least a 10 / 90 split
            float p = (split_success > 0) ? (float)meshA.n_rows / (float)n_sub_faces : 0.5f;
            split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

            // Attempt to split along another axis if failed for longest axis
            int first_test = std::abs(split_success);
            if (split_success < 0) // Failed condition
            {
                // Test second axis
                if (first_test == 2 || first_test == 3) // Test x-axis
                    split_success = quadriga_lib::triangle_mesh_split(*sub_mesh_it, meshA, meshB, 1, &split_ind);
                else // Test y-axis
                    split_success = quadriga_lib::triangle_mesh_split(*sub_mesh_it, meshA, meshB, 2, &split_ind);

                p = (split_success > 0) ? (float)meshA.n_rows / (float)n_sub_faces : 0.5f;
                split_success = (p < 0.1f || p > 0.9f) ? -split_success : split_success;

                // If we still failed, we test the third axis
                if ((first_test == 1 && split_success == -2) || (first_test == 2 && split_success == -1))
                    split_success = quadriga_lib::triangle_mesh_split(*sub_mesh_it, meshA, meshB, 3, &split_ind);
                else if (first_test == 3 && split_success == -1)
                    split_success = quadriga_lib::triangle_mesh_split(*sub_mesh_it, meshA, meshB, 2, &split_ind);

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
    sub_mesh_ind.set_size(n_sub);
    unsigned *p_sub_ind = sub_mesh_ind.memptr();

    for (arma::uword i_sub = 0; i_sub < n_sub; ++i_sub)
    {
        arma::uword n_sub_faces = c[i_sub].n_rows;
        arma::uword n_align = (n_sub_faces % vec_size == 0) ? n_sub_faces : (n_sub_faces / vec_size + 1) * vec_size;
        p_sub_ind[i_sub] = (unsigned)n_out;
        n_out += n_align;
    }

    // Assemble output
    mesh_out.set_size(n_out, 9);
    dtype *p_mesh_out = mesh_out.memptr();

    const arma::uword *p_mtl_in = process_mtl_ind ? mtl_ind.memptr() : nullptr;
    arma::uword *p_mtl_out = nullptr;
    if (process_mtl_ind)
    {
        mtl_ind_out->set_size(n_out);
        p_mtl_out = mtl_ind_out->memptr();
    }

    unsigned *p_mesh_index = nullptr;
    if (mesh_ind != nullptr)
    {
        mesh_ind->zeros(n_out);
        p_mesh_index = mesh_ind->memptr();
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

        // Copy material indices
        if (process_mtl_ind)
        {
            arma::uword offset_out = (arma::uword)p_sub_ind[i_sub];
            for (arma::uword i_sub_face = 0; i_sub_face < n_sub_faces; ++i_sub_face)
                p_mtl_out[offset_out + i_sub_face] = p_mtl_in[pi[i_sub_face]];
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
            arma::Mat<dtype> aabb = quadriga_lib::triangle_mesh_aabb(c[i_sub]);
            dtype *p_box = aabb.memptr();

            dtype x = p_box[0] + (dtype)0.5 * (p_box[1] - p_box[0]);
            dtype y = p_box[2] + (dtype)0.5 * (p_box[3] - p_box[2]);
            dtype z = p_box[4] + (dtype)0.5 * (p_box[5] - p_box[4]);

            arma::uword i_start = (arma::uword)p_sub_ind[i_sub] + n_sub_faces;
            arma::uword i_end = (i_sub == n_sub - 1) ? n_out : (arma::uword)p_sub_ind[i_sub + 1];

            // Geometry padding: dummy triangle collapsed to the AABB center
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
                }

            // Material padding: index 0 (transparent fallback / air)
            if (process_mtl_ind)
                for (arma::uword i_pad = i_start; i_pad < i_end; ++i_pad)
                    p_mtl_out[i_pad] = 0;
        }
    }

    return n_sub;
}

template arma::uword quadriga_lib::triangle_mesh_segmentation(const arma::Mat<float> &mesh, arma::Mat<float> &mesh_out,
                                                              arma::u32_vec &sub_mesh_ind, arma::uword target_size, arma::uword vec_size,
                                                              const arma::uvec &mtl_ind, arma::uvec *mtl_ind_out, arma::u32_vec *mesh_ind);

template arma::uword quadriga_lib::triangle_mesh_segmentation(const arma::Mat<double> &mesh, arma::Mat<double> &mesh_out,
                                                              arma::u32_vec &sub_mesh_ind, arma::uword target_size, arma::uword vec_size,
                                                              const arma::uvec &mtl_ind, arma::uvec *mtl_ind_out, arma::u32_vec *mesh_ind);
