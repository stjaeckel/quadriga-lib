// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "quadriga_lib.hpp"
#include "qd_helpers_python.hpp"



/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# point_cloud_aabb
Compute the axis-aligned bounding boxes (AABB) of a 3D point cloud

- Each row of the output contains `{x_min, x_max, y_min, y_max, z_min, z_max}` for one sub-cloud
- If `sub_cloud_ind` is empty or omitted, the entire input is treated as a single cloud; last
  index spans to end of `points`
- Output row count is zero-padded to the nearest multiple of `vec_size`; padding rows are zeros

## Usage:
```
aabb = quadriga_lib.RTtools.point_cloud_aabb( points, sub_cloud_ind, *, vec_size )
```

## Input Arguments:
- **`points`** — 3D point coordinates; `(n_points, 3)`
- **`sub_cloud_ind`** *(optional)* — 0-based row indices marking the start of each sub-cloud;
  use [[point_cloud_segmentation]] to generate; uint32; `(n_sub,)`; default: `None`
- **`vec_size`** *(optional)* — SIMD alignment padding factor (e.g. 4, 8, 16); default: 1

## Output Argument:
- **`aabb`** — Bounding box matrix; `(n_out, 6)` where `n_out` is `n_sub` padded to a multiple of `vec_size`

## See also:
- [[point_cloud_segmentation]] (generate sub-cloud indices)
- [[ray_point_intersect]] (use AABBs for intersection)
MD!*/

py::array_t<double> point_cloud_aabb(const py::array_t<double> &points,
                                     const py::handle &sub_cloud_ind,
                                     arma::uword vec_size)

{
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);
    const auto sub_cloud_ind_arma = qd_python_numpy2arma_Col<unsigned>(sub_cloud_ind, true, false, "sub_cloud_ind");

    const arma::mat aabb = quadriga_lib::point_cloud_aabb(points_arma, sub_cloud_ind_arma, vec_size);

    return qd_python_copy2numpy(&aabb);
}

/*!MD
# point_cloud_segmentation
Reorganize a point cloud into spatial sub-clouds for efficient processing

- Recursively partitions a 3D point cloud into sub-clouds by splitting along bounding box axes at the midpoint
- Sub-clouds can be padded to a multiple of `vec_size` for SIMD alignment; padding points are placed at the sub-cloud AABB center
- Produces a reorganized point array and index maps to track reordering

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.point_cloud_segmentation( points, *, target_size, vec_size )

# Unpacked outputs
points_out, sub_cloud_ind, forward_ind, reverse_ind = \
    quadriga_lib.RTtools.point_cloud_segmentation( points, *, target_size, vec_size )
```

## Inputs:
- **`points`** — Original 3D point cloud; `(n_points, 3)`
- **`target_size`** *(optional)* — Maximum points per sub-cloud before padding; default: 1024
- **`vec_size`** *(optional)* — SIMD/CUDA alignment; sub-cloud size is padded to a multiple of
  this value; no padding when `1`; `ray_point_intersect` accepts any value, padding (e.g. 8 for AVX2) only
  improves its speed; default: 1

## Outputs:
- **`points_out`** — Reorganized point cloud with points grouped by sub-cloud; `(n_points_out, 3)`
- **`sub_cloud_ind`** — 0-based starting index of each sub-cloud within `points_out`; uint32; `(n_sub,)`
- **`forward_ind`** *(optional)* — 1-based row in `points` of each row of `points_out`; padding entries are `0`; uint32; `(n_points_out,)`
- **`reverse_ind`** *(optional)* — 0-based row in `points_out` of each row of `points`; uint32; `(n_points,)`
MD!*/

py::tuple point_cloud_segmentation(const py::array_t<double> &points,
                                   arma::uword target_size,
                                   arma::uword vec_size)
{
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);

    arma::mat points_out;
    arma::u32_vec sub_cloud_ind, forward_ind, reverse_ind;

    quadriga_lib::point_cloud_segmentation<double>(points_arma, points_out, sub_cloud_ind,
                                                   target_size, vec_size, &forward_ind, &reverse_ind);

    auto points_out_p = qd_python_copy2numpy(&points_out);
    auto sub_cloud_ind_p = qd_python_copy2numpy(&sub_cloud_ind);
    auto forward_ind_p = qd_python_copy2numpy(&forward_ind);
    auto reverse_ind_p = qd_python_copy2numpy(&reverse_ind);

    return py::make_tuple(points_out_p, sub_cloud_ind_p, forward_ind_p, reverse_ind_p);
}

/*!MD
# triangle_mesh_aabb
Calculate the axis-aligned bounding box (AABB) of a triangle mesh and its sub-meshes

- Computes the AABB for each sub-mesh; used to accelerate ray tracing by cheaply excluding non-intersecting geometry
- Each triangle row: `{x1, y1, z1, x2, y2, z2, x3, y3, z3}`
- Output columns: `{x_min, x_max, y_min, y_max, z_min, z_max}`
- If `vec_size > 1`, output rows are padded to the next multiple of `vec_size`

## Usage:
```
aabb = quadriga_lib.RTtools.triangle_mesh_aabb( mesh, sub_mesh_ind, *, vec_size )
```

## Inputs:
- **`mesh`** — Triangle mesh vertices in global Cartesian coordinates; further columns are ignored; `(n_triangles, 9)`
- **`sub_mesh_ind`** *(optional)* — 0-based start indices of sub-meshes; if omitted, the AABB
  of the entire mesh is returned; uint32; `(n_sub,)`; default: `None`
- **`vec_size`** *(optional)* — Alignment size for SIMD/CUDA padding (e.g., `8` for AVX2, `32` for CUDA); default: 1

## Output:
- **`aabb`** — Axis-aligned bounding boxes, one row per sub-mesh; `(n_sub_aligned, 6)`

## See also:
- [[triangle_mesh_segmentation]] (for calculating sub-meshes)
MD!*/

py::array_t<double> triangle_mesh_aabb(const py::array_t<double> &mesh,
                                       const py::handle &sub_mesh_ind,
                                       arma::uword vec_size)
{
    const auto triangles_arma = qd_python_numpy2arma_Mat(mesh, true);
    const auto sub_mesh_index_arma = qd_python_numpy2arma_Col<unsigned>(sub_mesh_ind, true, false, "sub_mesh_ind");

    const arma::mat aabb = quadriga_lib::triangle_mesh_aabb(triangles_arma, sub_mesh_index_arma, vec_size);

    return qd_python_copy2numpy(&aabb);
}

/*!MD
# triangle_mesh_segmentation
Reorganize a 3D triangular mesh into spatially clustered sub-meshes for faster processing

- Recursively partitions mesh by axis-aligned bounding box until each sub-mesh contains no more
  than `target_size` triangles
- Output mesh retains all original triangles but in reordered sequence; sub-meshes are padded with
  zero-sized dummy triangles to align row counts to `vec_size`
- Dummy triangles are placed at the AABB center of their sub-mesh; real entries in `mesh_ind` are
  1-based, with 0 reserved to mark padding entries
- If `mtl_ind` is provided, material indices are reordered and padded in the same way; padding
  entries get index 0

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.triangle_mesh_segmentation( mesh, *, target_size, vec_size, mtl_ind )

# Unpacked outputs
mesh_out, sub_mesh_ind, mesh_ind, mtl_ind_out = \
    quadriga_lib.RTtools.triangle_mesh_segmentation( mesh, *, target_size, vec_size, mtl_ind )
```

## Inputs:
- **`mesh`** — Triangle vertices, each row `[x1,y1,z1,x2,y2,z2,x3,y3,z3]`; `(n_mesh, 9)`
- **`target_size`** — Target triangle count per sub-mesh; for best performance set near sqrt(n_mesh); default: 1024
- **`vec_size`** — SIMD/GPU alignment size (e.g. 8 for AVX2, 32 for CUDA); each sub-mesh row count
  is rounded up to a multiple of this value; default: 1
- **`mtl_ind`** — 1-based material index per face (0 = no material; the `mtl_ind`/`csv_ind` output of
  [[obj_file_read]]); `(n_mesh,)` or `None`; default: `None`

## Outputs:
- **`mesh_out`** — Reordered and padded triangle vertices; `(n_mesh_out, 9)`
- **`sub_mesh_ind`** — 0-based start indices of sub-meshes in `mesh_out`; uint32; `(n_sub,)`
- **`mesh_ind`** — 1-based row in the input mesh of each output triangle (0 = padding); uint32; `(n_mesh_out,)`
- **`mtl_ind_out`** — Reordered and padded material indices; `(n_mesh_out,)`; empty if `mtl_ind` is not provided
MD!*/

py::tuple triangle_mesh_segmentation(const py::array_t<double> &mesh,
                                     arma::uword target_size,
                                     arma::uword vec_size,
                                     const py::handle &mtl_ind)
{
    const auto triangles_a = qd_python_numpy2arma_Mat<double>(mesh, true);
    const auto mtl_ind_a = qd_python_numpy2arma_Col<arma::uword>(mtl_ind, true, false, "mtl_ind");

    arma::mat mesh_out;
    arma::uvec mtl_ind_out;
    arma::u32_vec sub_mesh_ind, mesh_ind;

    arma::uvec *p_mtl_ind_out = mtl_ind_a.is_empty() ? nullptr : &mtl_ind_out;

    quadriga_lib::triangle_mesh_segmentation<double>(triangles_a, mesh_out, sub_mesh_ind,
                                                     target_size, vec_size,
                                                     mtl_ind_a, p_mtl_ind_out, &mesh_ind);

    return py::make_tuple(
        qd_python_copy2numpy(&mesh_out),
        qd_python_copy2numpy<unsigned, py::ssize_t>(&sub_mesh_ind),
        qd_python_copy2numpy<unsigned, py::ssize_t>(&mesh_ind),
        qd_python_copy2numpy<arma::uword, py::ssize_t>(&mtl_ind_out));
}
