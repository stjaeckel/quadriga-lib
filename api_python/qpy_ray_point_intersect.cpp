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

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the origin, enabling energy spread simulation
- Returns, for each point, the list of ray indices whose beam intersects that point
- All internal computations use single precision

## Usage:
```
hit_count, ray_ind = quadriga_lib.RTtools.ray_point_intersect( orig, trivec, tridir, points, sub_cloud_ind, use_kernel, gpu_id )
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
  3 = CUDA; throws if unavailable; auto mode selects CUDA when `n_points >= 10000` and CUDA is
  available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA; default: 0

## Outputs:
- **`hit_count`** — Number of beams intersecting each point; uint32; `(n_points,)`
- **`ray_ind`** — List of length `(n_points,)`; each list entry is a 1-D array of 0-based ray indices
  that hit that point. Entries may be empty if no hit was detected; uint32

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

    arma::u32_vec hit_count;
    auto hit_count_p = qd_python_init_output(points_arma.n_rows, &hit_count);

    auto ray_ind = quadriga_lib::ray_point_intersect<double>(&points_arma, &orig_arma, &trivec_arma, &tridir_arma,
                                                             &sub_cloud_ind_arma, &hit_count, use_kernel, gpu_id);

    auto index_p = qd_python_copy2list<arma::u32_vec, py::ssize_t>(&ray_ind);

    return py::make_tuple(hit_count_p, index_p);
}