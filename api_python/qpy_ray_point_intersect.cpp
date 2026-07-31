// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "python_arma_adapter.hpp"
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
- Results use a compressed sparse row layout: `hit_index` holds the ray indices grouped by point and
  `hit_offset` marks where each point's block starts
- The rays hitting point `i` are `hit_index[hit_offset[i]:hit_offset[i+1]]`, a zero-copy numpy view
- `numpy.split(hit_index, hit_offset[1:-1])` produces a per-point list of arrays if that form is needed
- All internal computations use single precision

## Usage:
```
hit_count, hit_index, hit_offset = quadriga_lib.RTtools.ray_point_intersect( orig, trivec, tridir,
    points, sub_cloud_ind, use_kernel, gpu_id )
```

## Inputs:
- **`orig`** — Ray origin positions in global Cartesian coordinates; `(n_ray, 3)`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order
  `{v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z}`; `(n_ray, 9)`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates; not normalized;
  order `{d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z}`; `(n_ray, 9)`
- **`points`** — 3D point cloud coordinates; `(n_points, 3)`
- **`sub_cloud_index`** *(optional)* — 0-based segment boundary indices for the point cloud
  (see `quadriga_lib.point_cloud_segmentation`); uint32; `(n_sub,)`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2,
  3 = CUDA; throws if unavailable; auto mode selects CUDA when `n_points >= 500` and CUDA is
  available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA; default: 0

## Outputs:
- **`hit_count`** — Number of beams intersecting each point; equals `numpy.diff(hit_offset)`; uint32;
  `(n_points,)`
- **`hit_index`** — Flat list of 0-based ray indices, grouped by point; `(n_hit,)`
- **`hit_offset`** — Start of each point's block within `hit_index`; the last element equals `n_hit`;
  uint32; `(n_points + 1,)`

## See also:
- [[icosphere]] (for generating beams)
- [[point_cloud_segmentation]] (for generating point cloud segments)
- [[ray_triangle_intersect]] (for calculating intersection of rays and triangles)
MD!*/

py::tuple ray_point_intersect(const py::array_t<double> &orig,
                              const py::array_t<double> &trivec,
                              const py::array_t<double> &tridir,
                              const py::array_t<double> &points,
                              const py::array_t<unsigned> &sub_cloud_ind,
                              int use_kernel,
                              int gpu_id)
{

    const auto orig_arma = qd_python_numpy2arma_Mat(orig, true);
    const auto trivec_arma = qd_python_numpy2arma_Mat(trivec, true);
    const auto tridir_arma = qd_python_numpy2arma_Mat(tridir, true);
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);
    const auto sub_cloud_ind_arma = qd_python_numpy2arma_Col(sub_cloud_ind, true);

    // Both sizes are known up front, so the library writes straight into the
    // numpy buffers. Sizing them exactly prevents a resize inside the library,
    // which would detach the Armadillo object from the numpy memory.
    arma::u32_vec hit_count, hit_offset;
    auto hit_count_p = qd_python_init_output(points_arma.n_rows, &hit_count);
    auto hit_offset_p = qd_python_init_output(points_arma.n_rows + 1, &hit_offset);

    // The flat hit list is sized by the library, so it cannot be pre-allocated
    std::vector<unsigned> hit_index;

    quadriga_lib::ray_point_intersect<double>(points_arma, orig_arma, trivec_arma, tridir_arma,
                                              &hit_index, &hit_offset, &hit_count, nullptr,
                                              &sub_cloud_ind_arma, use_kernel, gpu_id);

    // Copy the flat list into numpy, widening to the signed 64 bit index type.
    // The Armadillo wrapper only aliases the vector, it does not copy.
    py::array_t<py::ssize_t> hit_index_p;
    if (!hit_index.empty())
    {
        const arma::u32_vec hit_index_arma(hit_index.data(), (arma::uword)hit_index.size(), false, true);
        hit_index_p = qd_python_copy2numpy<unsigned, py::ssize_t>(&hit_index_arma);
    }

    return py::make_tuple(hit_count_p, hit_index_p, hit_offset_p);
}