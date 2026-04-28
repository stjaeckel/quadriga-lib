// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- Internal computations always use single precision for AVX2 and CUDA kernels; only GENERIC has `double` support

## Usage:
```
# Output as tuple
data = RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id )

# Unpacked outputs
fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id )
```

## Inputs:
- **`orig`** — Ray origins in GCS; `(n_ray, 3)`
- **`dest`** — Ray destinations in GCS; `(n_ray, 3)`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `(n_mesh, 9)`
- **`sub_mesh_index`** (optional) — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; 0-based; uint32; `(n_sub)`
- **`aabb`** (optional) — Pre-computed axis-aligned bounding boxes per sub-mesh; each row:
  `{x_min x_max y_min y_max z_min z_max}`; if empty or omitted, AABBs are computed from `mesh`; `(n_sub, 6)`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA;
  throws if unavailable; auto mode selects CUDA when `n_ray >= 10000` and CUDA is available, else AVX2,
  else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

## Outputs:
- **`fbs`** — First-bounce intersection points in GCS; `(n_ray, 3)`
- **`sbs`** — Second-bounce intersection points in GCS; `(n_ray, 3)`
- **`no_interact`** — Total number of intersections per ray between `orig` and `dest`; uint32; `(n_ray,)`
- **`fbs_ind`** — 1-based index of first intersected mesh element; 0 = none; uint32; `(n_ray,)`
- **`sbs_ind`** — 1-based index of second intersected mesh element; 0 = none; uint32; `(n_ray,)`

## See also:
- [[obj_file_read]] (load mesh from OBJ file)
- [[triangle_mesh_segmentation]] (compute sub-mesh indices)
- [[triangle_mesh_aabb]] (compute AABBs)
- [[ray_point_intersect]] (beam interactions with sampling points)
- [[icosphere]] (generate ray beams)
MD!*/

py::tuple ray_triangle_intersect(const py::array_t<double> &orig,
                                 const py::array_t<double> &dest,
                                 const py::array_t<double> &mesh,
                                 const py::array_t<unsigned> &sub_mesh_index,
                                 const py::array_t<double> &aabb,
                                 int use_kernel,
                                 int gpu_id)
{
    const auto orig_arma = qd_python_numpy2arma_Mat(orig, true);
    const auto dest_arma = qd_python_numpy2arma_Mat(dest, true);
    const auto triangles_arma = qd_python_numpy2arma_Mat(mesh, true);
    const auto sub_mesh_index_arma = qd_python_numpy2arma_Col(sub_mesh_index, true);
    const auto aabb_arma = qd_python_numpy2arma_Mat(aabb, true);

    arma::uword n_ray = orig_arma.n_rows;

    // Pre-allocate outputs in Python memory and map Armadillo wrappers to them
    arma::mat fbs, sbs;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;

    auto fbs_p = qd_python_init_output(n_ray, (arma::uword)3, &fbs);
    auto sbs_p = qd_python_init_output(n_ray, (arma::uword)3, &sbs);
    auto no_interact_p = qd_python_init_output(n_ray, &no_interact);
    auto fbs_ind_p = qd_python_init_output(n_ray, &fbs_ind);
    auto sbs_ind_p = qd_python_init_output(n_ray, &sbs_ind);

    // Resolve optional pointers
    const arma::u32_vec *p_sub_mesh_index = sub_mesh_index_arma.empty() ? nullptr : &sub_mesh_index_arma;
    const arma::mat *p_aabb = aabb_arma.empty() ? nullptr : &aabb_arma;

    quadriga_lib::ray_triangle_intersect<double>(&orig_arma, &dest_arma, &triangles_arma,
                                                 &fbs, &sbs, &no_interact, &fbs_ind, &sbs_ind,
                                                 p_sub_mesh_index, p_aabb, use_kernel, gpu_id);

    return py::make_tuple(fbs_p, sbs_p, no_interact_p, fbs_ind_p, sbs_ind_p);
}