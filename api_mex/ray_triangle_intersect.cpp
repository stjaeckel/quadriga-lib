// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_tools.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# RAY_TRIANGLE_INTERSECT
Compute ray-triangle intersections in 3D using the Möller–Trumbore algorithm

- Counts the total number of intersections between `orig` and `dest`
- Computes the coordinates and object IDs of the first two intersections per ray (FBS/SBS)
- Internal computations always use single precision for AVX2 and CUDA kernels; only GENERIC has `double` support

## Usage:
```
[ fbs, sbs, no_interact, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( ...
    orig, dest, mesh, sub_mesh_index, aabb, use_kernel, gpu_id );
```

## Inputs:
- **`orig`** — Ray origins in GCS; `[n_ray, 3]`
- **`dest`** — Ray destinations in GCS; `[n_ray, 3]`
- **`mesh`** — Triangular mesh; each row: `{x1 y1 z1 x2 y2 z2 x3 y3 z3}`; `[n_mesh, 9]`
- **`sub_mesh_index`** (optional) — Start indices of sub-meshes in `mesh`; enables AABB-accelerated
  traversal; 1-based; `[n_sub]`
- **`aabb`** (optional) — Pre-computed axis-aligned bounding boxes per sub-mesh; each row:
  `{x_min x_max y_min y_max z_min z_max}`; if empty or omitted, AABBs are computed from `mesh`; `[n_sub, 6]`
- **`use_kernel`** *(optional)* — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA;
  throws if unavailable; auto mode selects CUDA when `n_ray >= 10000` and CUDA is available, else AVX2,
  else GENERIC.
- **`gpu_id`** *(optional)* — CUDA device ID; ignored when not using CUDA

## Outputs:
- **`fbs`** — First-bounce intersection points in GCS; `[n_ray, 3]`
- **`sbs`** — Second-bounce intersection points in GCS; `[n_ray, 3]`
- **`no_interact`** — Total number of intersections per ray between `orig` and `dest`; uint32; `[n_ray]`
- **`fbs_ind`** — 1-based index of first intersected mesh element; 0 = none; uint32; `[n_ray]`
- **`sbs_ind`** — 1-based index of second intersected mesh element; 0 = none; uint32; `[n_ray]`

## See also:
- [[obj_file_read]] (load mesh from OBJ file)
- [[triangle_mesh_segmentation]] (compute sub-mesh indices)
- [[triangle_mesh_aabb]] (compute AABBs)
- [[ray_point_intersect]] (beam interactions with sampling points)
- [[icosphere]] (generate ray beams)
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

    // Optional inputs
    arma::u32_vec sub_mesh_index = (nrhs < 4) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[3], true);
    arma::mat aabb = (nrhs < 5) ? arma::mat() : qd_mex_get_Mat<double>(prhs[4]);
    int use_kernel = (nrhs < 6) ? 0 : qd_mex_get_scalar<int>(prhs[5], "use_kernel", 0);
    int gpu_id = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "gpu_id", 0);

    // Convert sub_mesh_index to 0-based
    for (unsigned &val : sub_mesh_index)
    {
        if (val == 0)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Values in 'sub_mesh_index' cannot be 0.");
        --val;
    }

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