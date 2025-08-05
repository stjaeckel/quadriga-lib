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

#include "python_arma_adapter.hpp"
#include "quadriga_lib.hpp"

/*!SECTION
Site-Specific Simulation Tools
SECTION!*/

/*!MD
# ray_point_intersect
Calculates which 3D points are intersected by volumetric ray beams

## Description:
Unlike traditional ray tracing (rays are infinitesimal lines), **beam tracing** models rays as
volumes. Each beam is defined by a triangular wavefront whose three vertices diverge as the
beam propagates, capturing real-world spread (e.g., radio-wave divergence) and enabling realistic
energy distribution across the beam’s cross-section. Because beams have volume, intersection tests
are volumetric rather than line-to-geometry.<br><br>

A ray beam is specified by:
- An origin point.
- Three aperture vectors from the origin to the vertices of an initial triangular wavefront
  that defines the beam’s cross-section at the origin.
- Three per-vertex direction vectors (one per vertex) that govern how each vertex, and thus the
  triangle, diverges as the beam extends. Directions need not be normalized.<br><br>

What the function does
- Tests whether each point in a 3D Cartesian point cloud lies inside any of the defined ray beams.
- For every input point, returns a list of 0-based ray indices of beams that intersect that point.<br><br>

Performance & usage notes
- Optional support for pre-segmented point clouds (e.g., from [[point_cloud_segmentation]]) to reduce computation.
- All internal computations use single-precision floats for speed.
- Utilizes AVX2 vectorization when supported by the CPU.
- For best accuracy, use a small tube radius and well-distributed points.

## Usage:
```
from quadriga_lib import RTtools

hit_count, ray_ind = RTtools.ray_point_intersect( orig, trivec, tridir, points, sub_cloud_ind, target_size )
```

## Input Arguments:
- **`orig`**<br>
  Ray origins in 3D Cartesian coordinates; Shape: `(n_ray, 3)`

- **`trivec`**<br>
  Three vectors from each ray’s origin to the vertices of the triangular propagation tube (beam);
  Order per row: [v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z]; Shape: `(n_ray, 9)`

- **`tridir`**<br>
  Directions of the three vertex rays in Cartesian coordinates. Normalization not required.
  Order per row: [d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z]; Shape: `(n_ray, 9)`

- **`points`**<br>
  3D Cartesian coordinates of the point cloud to test. Shape: `(n_points, 3)`

- **`sub_cloud_ind`** (optional)<br>
  0-based start indices of sub-clouds used to partition points for performance. If not provided
  (empty array), sub-clouds may be computed automatically. Passing a scalar 0 disables sub-cloud calculation.
  Shape: (n_sub_cloud,) (strictly increasing; typically starts with 0)

- **`target_size`** (optional)<br>
  Desired sub-cloud size used only if `sub_cloud_ind` is not given or empty (guides automatic segmentation).
  If it is not given (set to 0) and `sub_cloud_ind` is empty or also not given, the optimal valued is computed 
  from the number of points.

## Output Arguments:
- **`hit_count`**<br>
  Number of ray beams that hit a point. Shape: `(n_points, )`

- **`ray_ind`**<br>
  List of length n_points; each list entry is a 1-D array of 0-based ray indices that hit that point.
  Entries may be empty if no hit was detected.

## See also:
- [[icosphere]] (for generating beams)
- [[point_cloud_segmentation]] (for generating point cloud segments)
- [[ray_triangle_intersect]] (for calculating intersection of rays and triangles)
MD!*/

py::tuple ray_point_intersect(const py::array_t<double> &orig,            // Ray origin points in GCS, Size [ n_ray, 3 ]
                              const py::array_t<double> &trivec,          // Vectors pointing from the origin to the vertices of the triangular propagation tube, Size [ n_ray, 9 ]
                              const py::array_t<double> &tridir,          // Directions of the vertex-rays; Cartesian format; Size [ n_ray, 9 ]
                              const py::array_t<double> &points,          // Points in 3D-Cartesian space; Size: [ n_points, 3 ]
                              const py::array_t<unsigned> &sub_cloud_ind, // Start indices of the sub-clouds in 0-based notation
                              arma::uword target_size)                    // Target value for the sub-cloud size
{

    const auto orig_arma = qd_python_numpy2arma_Mat(orig, true);
    const auto trivec_arma = qd_python_numpy2arma_Mat(trivec, true);
    const auto tridir_arma = qd_python_numpy2arma_Mat(tridir, true);
    const auto points_arma = qd_python_numpy2arma_Mat(points, true);
    auto sub_cloud_ind_arma = qd_python_numpy2arma_Col(sub_cloud_ind, false); // Create copy

    // Calculate optimal target size
    arma::uword n_points = points_arma.n_rows;
    if (target_size == 0)
    {
        target_size = 12 * (size_t)std::ceil(std::sqrt((double)n_points));
        target_size = (target_size < 1024) ? 0 : target_size;
    }

    // If sub_cloud_ind is not empty, use provided sub-clouds
    if (sub_cloud_ind_arma.n_elem != 0)
        target_size = 0;

    arma::u32_vec hit_count(n_points, arma::fill::zeros);
    std::vector<arma::u32_vec> index;

    if (target_size == 0) // Use provided sub-clouds
    {
        index = quadriga_lib::ray_point_intersect<double>(&points_arma, &orig_arma, &trivec_arma, &tridir_arma, &sub_cloud_ind_arma, &hit_count);
    }
    else // Calculate sub-clouds
    {
        arma::mat points_indexed;
        arma::u32_vec reverse_index;
        arma::u32_vec hit_count_local;

        quadriga_lib::point_cloud_segmentation<double>(&points_arma, &points_indexed, &sub_cloud_ind_arma, target_size, 8, nullptr, &reverse_index);
        index = quadriga_lib::ray_point_intersect(&points_indexed, &orig_arma, &trivec_arma, &tridir_arma, &sub_cloud_ind_arma, &hit_count_local);

        // Map to original order
        unsigned *p_hit_count = hit_count.memptr();
        unsigned *p_hit_count_local = hit_count_local.memptr();
        unsigned *i = reverse_index.memptr();
        for (arma::uword i_point = 0; i_point < n_points; ++i_point)
            p_hit_count[i_point] = p_hit_count_local[i[i_point]];
    }

    // Copy to python
    auto hit_count_p = qd_python_copy2numpy<arma::u32, ssize_t>(hit_count);
    auto index_p = qd_python_copy2numpy(index);

    return py::make_tuple(hit_count_p, index_p);
}