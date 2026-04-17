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

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

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
[ fbs, sbs, no_interact, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( ...
    orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id );
```

## Input Arguments:
- **`orig`**<br>
  Ray origins in global coordinate system (GCS). Size: `[ n_ray, 3 ]`

- **`dest`**<br>
  Ray destinations in GCS. Size: `[ n_ray, 3 ]`

- **`mesh`**<br>
  Triangular surface mesh. Each row contains the 3 vertices
  `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`. Size: `[ n_mesh, 9 ]`

- **`sub_mesh_index`** (optional)<br>
  Indexes indicating start of sub-meshes in `mesh`. Enables faster processing via segmentation.
  0-based; Type: uint32; Length: `[ n_sub ]`

- **`aabb`** (optional)<br>
  Pre-computed axis-aligned bounding boxes per sub-mesh. Each row contains
  `{x_min, x_max, y_min, y_max, z_min, z_max}`. Size: `[ n_sub, 6 ]`.
  If not provided, AABBs are computed internally from `mesh`.

- **`use_kernel`** (optional)<br>
  Selects the compute kernel: 0 = auto (default), 1 = GENERIC (scalar CPU), 2 = AVX2 (SIMD),
  3 = CUDA (GPU). An error is thrown if the requested kernel is not available at runtime.

- **`gpu_id`** (optional)<br>
  GPU device ID for CUDA kernel (default: 0). Ignored when not using CUDA.

## Output Arguments:
- **`fbs`**<br>
  First-bounce surface intersection points (FBS). Size: `[ n_ray, 3 ]`

- **`sbs`**<br>
  Second-bounce surface intersection points (SBS). Size: `[ n_ray, 3 ]`

- **`no_interact`**<br>
  Number of intersections per ray (0, 1, or 2); uint32; Length: `[ n_ray ]`

- **`fbs_ind`**<br>
  1-based index of the first intersected mesh element, 0 = no intersection; uint32; Length: `[ n_ray ]`

- **`sbs_ind`**<br>
  1-based index of the second intersected mesh element, 0 = no second intersection; uint32; Length: `[ n_ray ]`

## Caveat:
- Inputs can be provided in any numeric type; they are converted to double precision internally.
- The AVX2 and CUDA kernels always compute in single precision. Only the GENERIC kernel has full
  double precision support.
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 5)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments.");

    // Load inputs (cast to double if needed)
    arma::mat orig = qd_mex_get_Mat<double>(prhs[0]);
    arma::mat dest = qd_mex_get_Mat<double>(prhs[1]);
    arma::mat mesh = qd_mex_get_Mat<double>(prhs[2]);

    // Optional: sub_mesh_index
    arma::u32_vec sub_mesh_index;
    if (nrhs > 3 && !mxIsEmpty(prhs[3]))
    {
        if (mxIsUint32(prhs[3]))
            sub_mesh_index = qd_mex_reinterpret_Col<unsigned>(prhs[3]);
        else
            sub_mesh_index = qd_mex_typecast_Col<unsigned>(prhs[3]);
    }

    // Optional: aabb, use_kernel and gpu_id
    arma::mat aabb = (nrhs < 5) ? arma::mat() : qd_mex_get_Mat<double>(prhs[4]);
    int use_kernel = (nrhs < 6) ? 0 : qd_mex_get_scalar<int>(prhs[5], "use_kernel", 0);
    int gpu_id = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "gpu_id", 0);

    // Number of rays
    arma::uword n_rays = orig.n_rows;

    // Initialize output memory
    arma::mat fbs, sbs;
    arma::u32_vec no_interact, fbs_ind, sbs_ind;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&fbs, n_rays, 3);

    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&sbs, n_rays, 3);

    if (nlhs > 2)
        plhs[2] = qd_mex_init_output(&no_interact, n_rays);

    if (nlhs > 3)
        plhs[3] = qd_mex_init_output(&fbs_ind, n_rays);

    if (nlhs > 4)
        plhs[4] = qd_mex_init_output(&sbs_ind, n_rays);

    // Pointers
    arma::mat *p_fbs = fbs.empty() ? nullptr : &fbs;
    arma::mat *p_sbs = sbs.empty() ? nullptr : &sbs;
    arma::u32_vec *p_no_interact = no_interact.empty() ? nullptr : &no_interact;
    arma::u32_vec *p_fbs_ind = fbs_ind.empty() ? nullptr : &fbs_ind;
    arma::u32_vec *p_sbs_ind = sbs_ind.empty() ? nullptr : &sbs_ind;
    arma::u32_vec *p_sub_mesh_index = sub_mesh_index.empty() ? nullptr : &sub_mesh_index;
    arma::mat *p_aabb = aabb.empty() ? nullptr : &aabb;

    // Call library function
    CALL_QD(quadriga_lib::ray_triangle_intersect<double>(&orig, &dest, &mesh, p_fbs, p_sbs,
                                                         p_no_interact, p_fbs_ind, p_sbs_ind,
                                                         p_sub_mesh_index, p_aabb, use_kernel, gpu_id));
}