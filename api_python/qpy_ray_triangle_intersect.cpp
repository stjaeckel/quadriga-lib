// SPDX-License-Identifier: Apache-2.0
//
// quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
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
# RAY_TRIANGLE_INTERSECT
Calculates the intersection of rays and triangles in three dimensions

## Description:
- Implements the Möller–Trumbore algorithm to compute intersections between rays and triangles in 3D.
- Supports three compute kernels: **GENERIC** (scalar), **AVX2** (SIMD, 8 triangles in parallel), and **CUDA** (GPU).
- The `use_kernel` parameter selects the kernel: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA.
- In auto mode (0), CUDA is selected only when `n_ray >= 10000` and a CUDA-capable GPU is available;
  otherwise AVX2 is preferred if available, falling back to GENERIC.
- Can detect first and second intersections (FBS/SBS), number of intersections, and intersection indices.

## Usage:
```
from quadriga_lib import RTtools

# Output as tuple
data = RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id )

# Unpacked outputs
fbs, sbs, no_interact, fbs_ind, sbs_ind = RTtools.ray_triangle_intersect( orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id )
```

## Input Arguments:
- ndarray **`orig`**<br>
  Ray origins in global coordinate system (GCS); dtype: float64; shape: `(n_ray, 3)`

- ndarray **`dest`**<br>
  Ray destinations in GCS; dtype: float64; shape: `(n_ray, 3)`

- ndarray **`mesh`**<br>
  Triangular surface mesh. Each row contains the 3 vertices
  `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; dtype: float64; shape: `(n_mesh, 9)`

- ndarray **`sub_mesh_index`** (optional)<br>
  Indexes indicating start of sub-meshes in `mesh`. Enables faster processing via segmentation.
  0-based; dtype: uint32; shape: `(n_sub,)`

- ndarray **`aabb`** (optional)<br>
  Pre-computed axis-aligned bounding boxes per sub-mesh. Each row contains
  `{x_min, x_max, y_min, y_max, z_min, z_max}`; dtype: float64; shape: `(n_sub, 6)`.
  If not provided, AABBs are computed internally from `mesh`.

- int **`use_kernel`** (optional)<br>
  Selects the compute kernel: 0 = auto (default), 1 = GENERIC (scalar CPU), 2 = AVX2 (SIMD),
  3 = CUDA (GPU). An error is thrown if the requested kernel is not available at runtime.

- int **`gpu_id`** (optional)<br>
  GPU device ID for CUDA kernel (default: 0). Ignored when not using CUDA.

## Output Arguments:
- ndarray **`fbs`**, `data[0]`<br>
  First-bounce surface intersection points (FBS); dtype: float64; shape: `(n_ray, 3)`

- ndarray **`sbs`**, `data[1]`<br>
  Second-bounce surface intersection points (SBS); dtype: float64; shape: `(n_ray, 3)`

- ndarray **`no_interact`**, `data[2]`<br>
  Number of intersections per ray (0, 1, or 2); dtype: uint32; shape: `(n_ray,)`

- ndarray **`fbs_ind`**, `data[3]`<br>
  Index of the triangle that was hit by the ray at the FBS location; 1-based; 0 = no intersection;
  dtype: uint32; shape: `(n_ray,)`

- ndarray **`sbs_ind`**, `data[4]`<br>
  Index of the triangle that was hit by the ray at the SBS location; 1-based; 0 = no second
  intersection; dtype: uint32; shape: `(n_ray,)`

## Caveats:
- The AVX2 and CUDA kernels always compute in single precision. Only the GENERIC kernel has full
  double precision support.
- Zero-copy input mapping (no data duplication) is only used for Fortran-contiguous (column-major)
  NumPy arrays. C-contiguous inputs are silently copied and transposed.
- All outputs are Fortran-contiguous

## See also:
- [[obj_file_read]] (for loading mesh from an OBJ file)
- [[icosphere]] (for generating beams)
- [[triangle_mesh_segmentation]] (for calculating sub-meshes)
- [[ray_point_intersect]] (for calculating beam interactions with sampling points)
MD!*/

py::tuple ray_triangle_intersect(const py::array_t<double> &orig,             // Ray origin points in GCS, Size [ n_ray, 3 ]
                                 const py::array_t<double> &dest,             // Ray destination points in GCS, Size [ n_ray, 3 ]
                                 const py::array_t<double> &mesh,             // Faces of the triangular mesh (input), Size: [ n_mesh, 9 ]
                                 const py::array_t<unsigned> &sub_mesh_index, // Start indices of the sub-meshes in 0-based notation
                                 const py::array_t<double> &aabb,             // Pre-computed AABBs per sub-mesh, Size: [ n_sub, 6 ]
                                 int use_kernel,                              // Kernel selector: 0=auto, 1=GENERIC, 2=AVX2, 3=CUDA
                                 int gpu_id)                                  // GPU device ID for CUDA kernel
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