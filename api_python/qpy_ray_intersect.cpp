// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "qd_helpers_python.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# ray_point_intersect
Calculate intersections of ray beams with points in 3D space

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin,
  enabling energy spread simulation
- Reports, for each point, the list of ray indices whose beam intersects that point
- Results use a compressed sparse row layout: `hit_ind` holds the ray indices grouped by point and
  `hit_offset` marks where each point's block starts
- The rays hitting point `i` are `hit_ind[hit_offset[i]:hit_offset[i+1]]`, a zero-copy numpy view
- `numpy.split(hit_ind, hit_offset[1:-1])` produces a per-point list of arrays if that form is needed
- All internal computations use single precision

## Usage:
```
hit_count, hit_ind, hit_offset = quadriga_lib.RTtools.ray_point_intersect( points, orig, trivec, \
    tridir, sub_cloud_ind, *, use_kernel, gpu_id )
```

## Inputs:
- **`points`** — 3D point cloud coordinates; `(n_points, 3)`
- **`orig`** — Ray origin positions in global Cartesian coordinates; `(n_ray, 3)`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order
  `{v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z}`; `(n_ray, 9)`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates; not normalized;
  order `{d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z}`; `(n_ray, 9)`
- **`sub_cloud_ind`** — 0-based segment boundary indices for the point cloud
  (see `quadriga_lib.point_cloud_segmentation`); sub-clouds need not be aligned to the SIMD vector size (padding with
  `vec_size` is optional and only improves speed); `(n_sub,)`; default: `None`
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2,
  3 = CUDA; throws if unavailable and for other values; auto mode selects CUDA when `n_points >= 500` and CUDA is
  available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** — CUDA device ID; ignored when not using CUDA; default: 0

## Outputs:
- **`hit_count`** — Number of beams intersecting each point; equals `numpy.diff(hit_offset)`; uint32;
  `(n_points,)`
- **`hit_ind`** — Flat list of 0-based ray indices, grouped by point; int64; `(n_hit,)`
- **`hit_offset`** — Start of each point's block within `hit_ind`; the last element equals `n_hit`;
  uint32; `(n_points + 1,)`

## See also:
- [[icosphere]] (for generating beams)
- [[point_cloud_segmentation]] (for generating point cloud segments)
- [[ray_triangle_intersect]] (for calculating intersection of rays and triangles)
MD!*/

py::tuple ray_point_intersect(const py::array_t<double> &points,
                              const py::array_t<double> &orig,
                              const py::array_t<double> &trivec,
                              const py::array_t<double> &tridir,
                              const py::handle &sub_cloud_ind,
                              int use_kernel,
                              int gpu_id)
{

    const auto orig_arma = qd_python_numpy2arma_Mat(orig, true);
    const auto trivec_arma = qd_python_numpy2arma_Mat(trivec, true);
    const auto tridir_arma = qd_python_numpy2arma_Mat(tridir, true);
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);
    const auto sub_cloud_ind_arma = qd_python_numpy2arma_Col<unsigned>(sub_cloud_ind, true, false, "sub_cloud_ind");

    // Both sizes are known up front, so the library writes straight into the
    // numpy buffers. Sizing them exactly prevents a resize inside the library,
    // which would detach the Armadillo object from the numpy memory.
    arma::u32_vec hit_count, hit_offset;
    auto hit_count_p = qd_python_init_output(points_arma.n_rows, &hit_count);
    auto hit_offset_p = qd_python_init_output(points_arma.n_rows + 1, &hit_offset);

    // The flat hit list is sized by the library, so it cannot be pre-allocated
    std::vector<unsigned> hit_ind;

    quadriga_lib::ray_point_intersect<double>(points_arma, orig_arma, trivec_arma, tridir_arma,
                                              &hit_ind, &hit_offset, &hit_count, nullptr,
                                              sub_cloud_ind_arma, use_kernel, gpu_id);

    // Copy the flat list into numpy, widening to the signed 64 bit index type.
    // The Armadillo wrapper only aliases the vector, it does not copy.
    py::array_t<py::ssize_t> hit_index_p;
    if (!hit_ind.empty())
    {
        const arma::u32_vec hit_index_arma(hit_ind.data(), (arma::uword)hit_ind.size(), false, true);
        hit_index_p = qd_python_copy2numpy<unsigned, py::ssize_t>(&hit_index_arma);
    }

    return py::make_tuple(hit_count_p, hit_index_p, hit_offset_p);
}

/*!MD
# ray_triangle_intersect
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- The AVX2 and CUDA kernels compute in single precision, GENERIC in the precision of the input; with `double` input,
  the AVX2 kernel forms the vectors between ray origin, vertices and bounding boxes in `double` before rounding them, so
  its precision depends on the size of the scene, not on its distance from the coordinate origin (e.g. UTM coordinates);
  the CUDA kernel rounds the coordinates themselves, so its errors grow with the distance from the origin
- Triangle edges and vertices count as hits; hits at the same distance (e.g. on a shared edge) are reported in the
  order of the mesh index, the lower index first (all kernels)

## Usage:
```
# Output as tuple
data = quadriga_lib.RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_ind, aabb, *, use_kernel, gpu_id )

# Unpacked outputs
fbs, sbs, n_interact, fbs_ind, sbs_ind = quadriga_lib.RTtools.ray_triangle_intersect( orig, dest, mesh, \
    sub_mesh_ind, aabb, *, use_kernel, gpu_id )
```

## Inputs:
- **`orig`** — Ray origins in GCS; `(n_ray, 3)`
- **`dest`** — Ray destinations in GCS; `(n_ray, 3)`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; further columns are ignored; `(n_mesh, 9)`
- **`sub_mesh_ind`** — Start indices of sub-meshes in `mesh`; enables AABB-accelerated traversal; 0-based; `(n_sub,)`; default: `None`
- **`aabb`** — Pre-computed axis-aligned bounding boxes per sub-mesh; each row:
  `{x_min x_max y_min y_max z_min z_max}`; if empty or omitted, AABBs are computed from `mesh`; `(n_sub, 6)`; default: `None`
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA;
  throws if unavailable and for other values; auto mode selects CUDA when `n_ray >= 500` and CUDA is available, else AVX2,
  else GENERIC; default: 0
- **`gpu_id`** — CUDA device ID; ignored when not using CUDA; default: 0

## Outputs:
- **`fbs`** — First-bounce intersection points in GCS; `(n_ray, 3)`
- **`sbs`** — Second-bounce intersection points in GCS; `(n_ray, 3)`
- **`n_interact`** — Total number of intersections per ray between `orig` and `dest`; uint32; `(n_ray,)`
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
                                 const py::handle &sub_mesh_ind,
                                 const py::handle &aabb,
                                 int use_kernel,
                                 int gpu_id)
{
    const auto orig_arma = qd_python_numpy2arma_Mat(orig, true);
    const auto dest_arma = qd_python_numpy2arma_Mat(dest, true);
    const auto triangles_arma = qd_python_numpy2arma_Mat(mesh, true);
    const auto sub_mesh_index_arma = qd_python_numpy2arma_Col<unsigned>(sub_mesh_ind, true, false, "sub_mesh_ind");
    const auto aabb_arma = qd_python_numpy2arma_Mat<double>(aabb, true);

    arma::uword n_ray = orig_arma.n_rows;

    // Pre-allocate outputs in Python memory and map Armadillo wrappers to them
    arma::mat fbs, sbs;
    arma::u32_vec n_interact, fbs_ind, sbs_ind;

    auto fbs_p = qd_python_init_output(n_ray, (arma::uword)3, &fbs);
    auto sbs_p = qd_python_init_output(n_ray, (arma::uword)3, &sbs);
    auto n_interact_p = qd_python_init_output(n_ray, &n_interact);
    auto fbs_ind_p = qd_python_init_output(n_ray, &fbs_ind);
    auto sbs_ind_p = qd_python_init_output(n_ray, &sbs_ind);

    quadriga_lib::ray_triangle_intersect<double>(orig_arma, dest_arma, triangles_arma,
                                                 &fbs, &sbs, &n_interact, &fbs_ind, &sbs_ind,
                                                 sub_mesh_index_arma, aabb_arma, use_kernel, gpu_id);

    return py::make_tuple(fbs_p, sbs_p, n_interact_p, fbs_ind_p, sbs_ind_p);
}
