// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
// Part of quadriga-lib — see LICENSE for terms.

#include "mex.h"
#include "quadriga_lib.hpp"
#include "mex_helper_functions.hpp"

/*!SECTION
Site-specific simulation tools
SECTION!*/

/*!MD
# RAY_POINT_INTERSECT
Calculate intersections of ray beams with points in 3D space

- Models rays as volumetric beams defined by a triangular wavefront that diverges from the
  origin, enabling energy spread simulation
- Returns, for each point, the list of ray indices whose beam intersects that point
- All internal computations use single precision

## Usage:
```
[ hit_count, ray_ind ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, ...
    sub_cloud_index, use_kernel, gpu_id );
```

## Inputs:
- **`orig`** — Ray origin positions in global Cartesian coordinates; `[n_ray, 3]`
- **`trivec`** — Vectors from ray origin center to triangular wavefront vertices, order
  `{v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z}`; `[n_ray, 9]`
- **`tridir`** — Direction vectors of the three vertex-rays in Cartesian coordinates; not normalized;
  order `{d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z}`; `[n_ray, 9]`
- **`points`** — 3D point cloud coordinates; `[n_points, 3]`
- **`sub_cloud_index`** — Segment boundary indices for the point cloud; use [[point_cloud_segmentation]] to gnerate;
  uint32; `[n_sub]`; default: `[]` (not using sub-clouds)
- **`use_kernel`** — Compute kernel selector: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; throws if unavailable;
  auto mode selects CUDA when `n_points >= 500` and CUDA is available, else AVX2, else GENERIC; default: 0
- **`gpu_id`** — CUDA device ID; ignored when not using CUDA; default: 0

## Outputs:
- **`hit_count`** — Number of beams intersecting each point; `[n_points, 1]`
- **`ray_ind`** — Per-point list of 1-based ray indices that intersected that point; zero-padded to
  a regular 2D array (zero entries indicate unused slots); uint32; `[max_hits, n_points]`

## See also:
- [[icosphere]] (generate ray beams)
- [[point_cloud_segmentation]] (generate sub-cloud index)
- [[subdivide_rays]] (subdivide beams into sub-beams)
- [[ray_triangle_intersect]] (ray–triangle intersection)
- [[ray_mesh_interact]] (beam–mesh interaction)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate argument counts
    if (nrhs < 4 || nrhs > 7)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of output arguments.");

    // Read input data
    const arma::mat orig = qd_mex_get_Mat<double>(prhs[0]);
    const arma::mat trivec = qd_mex_get_Mat<double>(prhs[1]);
    const arma::mat tridir = qd_mex_get_Mat<double>(prhs[2]);
    const arma::mat points = qd_mex_get_Mat<double>(prhs[3]);
    arma::u32_vec sub_cloud_index = (nrhs < 5) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[4], true);
    const int use_kernel = (nrhs < 6) ? 0 : qd_mex_get_scalar<int>(prhs[5], "use_kernel", 0);
    const int gpu_id = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "gpu_id", 0);

    // Convert sub_mesh_index to 0-based
    for (unsigned &val : sub_cloud_index)
    {
        if (val == 0)
            mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Values in 'sub_cloud_index' cannot be 0.");
        --val;
    }

    // Wrap optional input pointer
    const arma::u32_vec *p_sub_cloud_index = sub_cloud_index.empty() ? nullptr : &sub_cloud_index;

    // hit_count output: size known (n_points), allocate MATLAB-owned memory if requested
    arma::u32_vec hit_count;
    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&hit_count, points.n_rows);

    arma::u32_vec *p_hit_count = hit_count.empty() ? nullptr : &hit_count;

    // ray_ind is assembled from the flat CSR list returned by the library: 'hit_index'
    // holds the ray indices grouped by point, 'hit_offset' marks the start of each
    // point's block. Both are sized by the C++ call, so the (nlhs > 1) guard is used.
    std::vector<unsigned> hit_index;
    arma::u32_vec hit_offset;
    std::vector<unsigned> *p_hit_index = (nlhs > 1) ? &hit_index : nullptr;
    arma::u32_vec *p_hit_offset = (nlhs > 1) ? &hit_offset : nullptr;

    // Call library function
    CALL_QD(quadriga_lib::ray_point_intersect<double>(points, orig, trivec, tridir,
                                                      p_hit_index, p_hit_offset, p_hit_count, nullptr,
                                                      p_sub_cloud_index, use_kernel, gpu_id));

    // Transpose the CSR list into the zero-padded [max_hits, n_points] MATLAB array.
    // qd_mex_init_output allocates through mxCreateNumericMatrix, so the unused slots
    // are already zero, and the 0-based to 1-based shift is folded into the write.
    // (Zero-padding acts as a "no ray" sentinel since valid indices are >= 1.)
    if (nlhs > 1)
    {
        const arma::uword n_points = points.n_rows;
        const unsigned *p_off = hit_offset.memptr();

        // The longest per-point list determines the number of rows
        arma::uword max_hits = 0;
        for (arma::uword i_point = 0; i_point < n_points; ++i_point)
        {
            arma::uword n_local = (arma::uword)(p_off[i_point + 1] - p_off[i_point]);
            max_hits = (n_local > max_hits) ? n_local : max_hits;
        }

        arma::u32_mat ray_ind;
        plhs[1] = qd_mex_init_output(&ray_ind, max_hits, n_points);

        if (max_hits != 0)
        {
            const unsigned *p_idx = hit_index.data();
            unsigned *p_ray_ind = ray_ind.memptr();

            for (arma::uword i_point = 0; i_point < n_points; ++i_point)
            {
                const unsigned first = p_off[i_point], last = p_off[i_point + 1];
                unsigned *p_col = p_ray_ind + unsigned(i_point * max_hits);

                for (unsigned k = first; k < last; ++k)
                    p_col[k - first] = p_idx[k] + 1u;
            }
        }
    }
}