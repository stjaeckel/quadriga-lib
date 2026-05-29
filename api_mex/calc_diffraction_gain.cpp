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
# CALC_DIFFRACTION_GAIN
Calculate diffraction gain for multiple TX-RX pairs using a 3D triangular mesh

- Estimates diffraction gain by evaluating Fresnel ellipsoid obstruction; each TX-RX path is divided
  into `n_path` elliptic-arc paths (controlled by `lod`), each approximated by `n_seg` line segments
- Segment attenuation is combined via weighted summation calibrated to 2D UTD coefficients,
  generalized to arbitrary 3D shapes
- Optional sub-mesh indexing (see [[triangle_mesh_segmentation]]) accelerates computation by skipping
  triangles whose bounding box does not intersect the TX-RX path

## Usage:
```
[ gain, coord ] = quadriga_lib.calc_diffraction_gain( orig, dest, mesh, mtl_prop, ...
    center_freq, lod, verbose, sub_mesh_index, use_kernel, gpu_id );
```

## Inputs:
- **`orig`** — TX positions; `[n_pos, 3]`
- **`dest`** — RX positions; `[n_pos, 3]`
- **`mesh`** — Triangle vertices, each row `{X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3}`; `[n_mesh, 9]`
- **`mtl_prop`** — Material properties; see [[obj_file_read]]; `[n_mesh, n_param]`
- **`center_freq`** — Center frequency
- **`lod`** — Level of detail (0–6), controls `n_path` and `n_seg`; see [[generate_diffraction_paths]]; default: 2
- **`verbose`** — Verbosity level; default: 0 (no output)
- **`sub_mesh_index`** — 1-based sub-mesh index for acceleration; see [[triangle_mesh_segmentation]];  `[n_mesh, 1]`;
  default: `[]` (not using sub-meshes)
- **`use_kernel`** — Kernel selection: 0 = auto, 1 = GENERIC, 2 = AVX2, 3 = CUDA; error if unavailable; default: 0
- **`gpu_id`** — CUDA device ID; ignored for non-CUDA kernels; default: 0

## Outputs:
- **`gain`** — Diffraction gain per TX-RX pair, linear scale; `[n_pos, 1]`
- **`coord`** — Diffracted path coordinates excluding endpoints; `[3, n_seg-1, n_pos]`

## See also:
- [[generate_diffraction_paths]] (controls path/segment count via `lod`)
- [[triangle_mesh_segmentation]] (generates `sub_mesh_index`)
- [[obj_file_read]] (defines `mtl_prop` format)
MD!*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 5 || nrhs > 10)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Wrong number of input arguments.");

    if (nlhs > 2)
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Too many output arguments.");

    // Load inputs (cast to double if needed)
    arma::mat orig = qd_mex_get_Mat<double>(prhs[0]);
    arma::mat dest = qd_mex_get_Mat<double>(prhs[1]);
    arma::mat mesh = qd_mex_get_Mat<double>(prhs[2]);
    arma::mat mtl_prop = qd_mex_get_Mat<double>(prhs[3]);

    double center_freq = qd_mex_get_scalar<double>(prhs[4], "center_frequency", 1.0e9);
    int lod = (nrhs < 6) ? 2 : qd_mex_get_scalar<int>(prhs[5], "lod", 2);
    int verbose = (nrhs < 7) ? 0 : qd_mex_get_scalar<int>(prhs[6], "verbose", 0);
    arma::u32_vec sub_mesh_index = (nrhs < 8) ? arma::u32_vec() : qd_mex_get_Col<unsigned>(prhs[7], true);

    if (!sub_mesh_index.empty() && arma::any(sub_mesh_index == 0))
        mexErrMsgIdAndTxt("quadriga_lib:CPPerror", "Entries in 'sub_mesh_index' cannot be 0 (1-based index).");
    else // Convert to 0-based
        sub_mesh_index -= 1;

    int use_kernel = (nrhs < 9) ? 0 : qd_mex_get_scalar<int>(prhs[8], "use_kernel", 0);
    int gpu_id = (nrhs < 10) ? 0 : qd_mex_get_scalar<int>(prhs[9], "gpu_id", 0);

    arma::uword n_pos = orig.n_rows;
    arma::uword n_seg = 0;
    if (lod == 1 || lod == 2)
        n_seg = 2;
    else if (lod == 3)
        n_seg = 3;
    else if (lod == 4)
        n_seg = 4;
    else if (lod == 5 || lod == 6)
        n_seg = 1;

    // Initialize output containers
    arma::vec gain;
    arma::cube coord;

    if (nlhs > 0)
        plhs[0] = qd_mex_init_output(&gain, n_pos);

    if (nlhs > 1)
        plhs[1] = qd_mex_init_output(&coord, 3, n_seg, n_pos);

    arma::vec *p_gain = gain.empty() ? nullptr : &gain;
    arma::cube *p_coord = coord.empty() ? nullptr : &coord;
    arma::u32_vec *p_sub_mesh_index = sub_mesh_index.empty() ? nullptr : &sub_mesh_index;

    CALL_QD(quadriga_lib::calc_diffraction_gain(&orig, &dest, &mesh, &mtl_prop,
                                                center_freq, lod, p_gain, p_coord, verbose,
                                                p_sub_mesh_index, use_kernel, gpu_id));
}